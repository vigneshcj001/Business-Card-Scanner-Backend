# main.py (FastAPI backend using OpenAI Vision for business card extraction)
import os
import io
import json
import base64
import logging
from datetime import datetime
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, File, UploadFile, Body, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import pytz

# Validation & enrichment libs (kept for phone/email/domain validation)
import phonenumbers
import tldextract
import validators

# OpenAI client (Responses API with vision)
from openai import OpenAI

# Load .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")
PHONE_DEFAULT_REGION = os.getenv("PHONE_DEFAULT_REGION", "IN")  # default phone region

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set — upload_card route will fail if used without the key.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI setup
app = FastAPI(title="Business Card OCR API (OpenAI Vision)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# -----------------------------------------
# JSON Encoder helper
# -----------------------------------------
class JSONEncoder:
    @staticmethod
    def encode(doc: Any):
        if isinstance(doc, ObjectId):
            return str(doc)
        if isinstance(doc, dict):
            return {k: JSONEncoder.encode(v) for k, v in doc.items()}
        if isinstance(doc, list):
            return [JSONEncoder.encode(x) for x in doc]
        return doc

# -----------------------------------------
# Pydantic models
# -----------------------------------------
class ContactCreate(BaseModel):
    name: Optional[str] = ""
    designation: Optional[str] = ""
    company: Optional[str] = ""
    phone_numbers: Optional[List[str]] = []
    email: Optional[str] = ""
    website: Optional[str] = ""
    address: Optional[str] = ""
    social_links: Optional[List[str]] = []
    more_details: Optional[str] = ""
    additional_notes: Optional[str] = ""

    @validator("phone_numbers", pre=True)
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            items = [x.strip() for x in v.split(",") if x.strip()]
            return items
        return v

    @validator("social_links", pre=True)
    def ensure_social_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            items = [x.strip() for x in v.split(",") if x.strip()]
            return items
        return v

    @validator("email", pre=True, always=True)
    def empty_or_valid_email(cls, v):
        v = (v or "").strip()
        if v == "":
            return ""
        if validators.email(v):
            return v
        raise ValueError("email must be a valid email address or empty")

# -----------------------------------------
# Utilities (validation, classification)
# -----------------------------------------
def now_ist() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

SOCIAL_PLATFORMS = {
    "linkedin": ["linkedin.com", "linkedin"],
    "twitter": ["twitter.com", "x.com", "t.co", "twitter"],
    "instagram": ["instagram.com", "instagr.am", "instagram"],
    "facebook": ["facebook.com", "fb.me", "facebook"],
    "telegram": ["t.me", "telegram.me", "telegram"],
    "whatsapp": ["wa.me", "whatsapp"]
}

COMPANY_KEYWORDS_LABELS = [
    ("Pvt Ltd", ["pvt ltd", "private limited", "pvt. ltd"]),
    ("Ltd", ["ltd", "limited"]),
    ("Inc", ["inc", "inc."]),
    ("LLP", ["llp"]),
    ("Solutions", ["solutions"]),
    ("Technologies", ["technologies", "tech"]),
    ("Works", ["works"])
]

ADDRESS_COUNTRY_KEYWORDS = {
    "India": ["india", "tamilnadu", "mumbai", "delhi", "bangalore", "chennai", "pincode", "pin"],
    "USA": ["usa", "united states", "california", "ny", "new york", "zip"],
    "UK": ["united kingdom", "england", "london", "uk"],
}

def _detect_social_platforms(links: List[str]) -> List[str]:
    found = set()
    for link in links or []:
        low = link.lower()
        for platform, tests in SOCIAL_PLATFORMS.items():
            if any(t in low for t in tests):
                found.add(platform)
    return sorted(found)

def _guess_company_type(company: str) -> str:
    if not company:
        return ""
    low = company.lower()
    for label, keywords in COMPANY_KEYWORDS_LABELS:
        for kw in keywords:
            if kw in low:
                return label
    return ""

def _guess_address_country(address: str) -> str:
    if not address:
        return ""
    low = address.lower()
    for country, keys in ADDRESS_COUNTRY_KEYWORDS.items():
        for k in keys:
            if k in low:
                return country
    return ""

def parse_phones(phone_list: List[str]) -> List[Dict[str, Any]]:
    out = []
    for raw in phone_list or []:
        candidate = re_sub_digits_plus_x(raw := str(raw))
        try:
            parsed = phonenumbers.parse(candidate, PHONE_DEFAULT_REGION)
            is_valid = phonenumbers.is_valid_number(parsed)
            e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164) if is_valid else candidate
            phone_type = None
            try:
                phone_type = phonenumbers.number_type(parsed).name
            except Exception:
                phone_type = None
            out.append({
                "raw": raw,
                "normalized": e164,
                "country_code": getattr(parsed, "country_code", None),
                "national_number": getattr(parsed, "national_number", None),
                "valid": bool(is_valid),
                "type": phone_type
            })
        except Exception:
            out.append({
                "raw": raw,
                "normalized": None,
                "country_code": None,
                "national_number": None,
                "valid": False,
                "type": None
            })
    return out

def re_sub_digits_plus_x(s: str) -> str:
    # keep digits, +, x, X for parse attempts
    import re
    return re.sub(r"[^\d\+xX]", "", s)

def classify_contact(contact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate & enrich contact. Leaves more_details empty unless provided by caller.
    """
    c = dict(contact)  # shallow copy

    phones = c.get("phone_numbers") or []
    if isinstance(phones, str):
        phones = [p.strip() for p in phones.split(",") if p.strip()]

    socials = c.get("social_links") or []
    if isinstance(socials, str):
        socials = [s.strip() for s in socials.split(",") if s.strip()]

    email = (c.get("email") or "").strip()
    website = (c.get("website") or "").strip()
    name = (c.get("name") or "").strip()
    company = (c.get("company") or "").strip()
    address = (c.get("address") or "").strip()
    notes = (c.get("additional_notes") or "").strip()
    designation = (c.get("designation") or "").strip()

    # Email validation
    email_valid = bool(validators.email(email)) if email else False

    # Website validation & domain
    website_valid = False
    domain = ""
    if website:
        website_try = website if website.startswith(("http://", "https://")) else ("http://" + website)
        website_valid = bool(validators.url(website_try))
        te = tldextract.extract(website)
        if te and te.domain:
            domain = ".".join([p for p in [te.domain, te.suffix] if p])

    # Phones parse
    phones_parsed = parse_phones(phones)

    # Social platform detection (include email + website)
    social_platforms = _detect_social_platforms(socials + [website] + [email])

    # Company type & address country
    company_type = _guess_company_type(company)
    address_country = _guess_address_country(address)

    # Name heuristics
    name_is_upper = bool(name and name.replace(" ", "").isupper())
    name_word_count = len(name.split()) if name else 0

    # Notes stats
    notes_len = len(notes)
    lines_in_notes = len([l for l in notes.splitlines() if l.strip()])

    # Build structured validation doc
    field_validations = {
        "email": {"value": email, "valid": email_valid},
        "website": {"value": website, "valid": website_valid, "domain": domain},
        "phones": phones_parsed,
        "social_platforms_detected": social_platforms,
        "company_type_guess": company_type,
        "address_country_hint": address_country,
        "name": {"value": name, "is_uppercase": name_is_upper, "word_count": name_word_count},
        "notes": {"length": notes_len, "lines": lines_in_notes},
        "designation": {"value": designation}
    }

    c["field_validations"] = field_validations

    if "more_details" not in c or not c.get("more_details"):
        c["more_details"] = ""

    c["phone_numbers"] = phones
    c["social_links"] = socials

    return c

# -----------------------------------------
# Routes
# -----------------------------------------
@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅ (OpenAI Vision)"}

# Upload route: uses OpenAI Responses API with image input
@app.post("/upload_card", status_code=status.HTTP_201_CREATED)
async def upload_card(file: UploadFile = File(...)):
    try:
        content = await file.read()

        # Prepare base64 data-uri for the image
        b64 = base64.b64encode(content).decode("utf-8")
        data_uri = f"data:{file.content_type};base64,{b64}"

        # Strict prompt asking for ONLY JSON (single object)
        prompt_text = (
            "You are a precise assistant that extracts structured information from a business card image.\n\n"
            "Return exactly ONE JSON object and NOTHING ELSE. The object must have these keys:\n"
            "  name (string), designation (string), company (string), phone_numbers (array of strings),\n"
            "  email (string), website (string), address (string), social_links (array of strings),\n"
            "  more_details (string), additional_notes (string)\n\n"
            "If a field is not present, return an empty string \"\" for strings and [] for lists.\n"
            "Do not include additional keys. Provide clean values without explanatory text.\n\n"
            "Now extract fields from the provided image. Provide only the JSON object.\n"
        )

        # Choose the vision-capable model you have access to (change as needed)
        model_name = "gpt-4o-mini-vision"  # change to the model available in your account

        # Call OpenAI Responses API (vision)
        resp = openai_client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": data_uri}
                    ]
                }
            ],
            temperature=0.0,
            max_output_tokens=1200,
        )

        # Extract textual output (robust to different response shapes)
        assistant_text = ""
        try:
            outputs = resp.output or []
            parts = []
            for part in outputs:
                ct = part.get("content")
                if isinstance(ct, list):
                    for c in ct:
                        if c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
                elif isinstance(ct, str):
                    parts.append(ct)
            assistant_text = "\n".join([p for p in parts if p]).strip()
        except Exception:
            assistant_text = getattr(resp, "output_text", "") or ""

        if not assistant_text:
            raise HTTPException(status_code=502, detail="OpenAI returned no textual output for OCR.")

        # Parse the assistant_text expecting a JSON object; extract first {...} if necessary
        def extract_first_json_object(s: str):
            start = s.find("{")
            if start == -1:
                raise ValueError("No JSON object found in model output.")
            depth = 0
            for i in range(start, len(s)):
                if s[i] == "{":
                    depth += 1
                elif s[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
            raise ValueError("Incomplete JSON object in model output.")

        try:
            json_str = assistant_text
            if not json_str.strip().startswith("{"):
                json_str = extract_first_json_object(assistant_text)
            extracted_json = json.loads(json_str)
        except Exception as e:
            logging.exception("Failed to parse JSON from OpenAI output.")
            raise HTTPException(status_code=502, detail=f"OpenAI response could not be parsed as JSON: {e}. Raw output (truncated): {assistant_text[:1000]}")

        # Ensure expected keys & normalise structure
        data = {
            "name": extracted_json.get("name", "") or "",
            "designation": extracted_json.get("designation", "") or "",
            "company": extracted_json.get("company", "") or "",
            "phone_numbers": extracted_json.get("phone_numbers", []) or [],
            "email": extracted_json.get("email", "") or "",
            "website": extracted_json.get("website", "") or "",
            "address": extracted_json.get("address", "") or "",
            "social_links": extracted_json.get("social_links", []) or [],
            "more_details": extracted_json.get("more_details", "") or "",
            "additional_notes": extracted_json.get("additional_notes", "") or ""
        }

        # Keep raw model output for debugging/troubleshooting
        data["_openai_raw"] = assistant_text
        data["_openai_model"] = model_name
        data["_processed_with_openai_at"] = now_ist()

        # classify & enrich (existing logic)
        data = classify_contact(data)

        # ensure more_details remains empty unless provided
        data["more_details"] = "" if not data.get("more_details") else data.get("more_details")
        data["created_at"] = now_ist()
        data["edited_at"] = ""

        result = collection.insert_one(data)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("upload_card (OpenAI) error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_card", status_code=status.HTTP_201_CREATED)
async def create_card(payload: ContactCreate):
    try:
        doc = payload.dict()
        doc = classify_contact(doc)
        if payload.more_details:
            doc["more_details"] = payload.more_details
        else:
            doc["more_details"] = ""
        doc["created_at"] = now_ist()
        doc["edited_at"] = ""
        if doc.get("email"):
            existing = collection.find_one({"email": doc["email"]})
            if existing:
                if not doc.get("more_details"):
                    doc["more_details"] = existing.get("more_details", "")
                doc["edited_at"] = now_ist()
                collection.update_one({"_id": existing["_id"]}, {"$set": doc})
                updated = collection.find_one({"_id": existing["_id"]})
                return {"message": "Updated existing contact", "data": JSONEncoder.encode(updated)}
        result = collection.insert_one(doc)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}
    except Exception as e:
        logging.exception("create_card error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find().sort([("_id", -1)]))
        return {"data": JSONEncoder.encode(docs)}
    except Exception as e:
        logging.exception("get_all_cards error")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        ts = now_ist()
        update_payload = {
            "additional_notes": payload.get("additional_notes", ""),
            "edited_at": ts
        }
        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_payload})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": JSONEncoder.encode(updated)}
    except Exception as e:
        logging.exception("update_notes error")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/update_card/{card_id}")
def update_card(card_id: str, payload: dict = Body(...)):
    try:
        allowed_fields = {
            "name", "designation", "company", "phone_numbers",
            "email", "website", "address", "social_links",
            "additional_notes", "more_details"
        }

        update_data = {k: v for k, v in payload.items() if k in allowed_fields}
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update.")

        existing = collection.find_one({"_id": ObjectId(card_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Card not found.")

        existing_more = existing.get("more_details", "")

        merged = dict(existing)
        merged.update(update_data)

        merged = classify_contact(merged)

        if "more_details" in update_data:
            merged["more_details"] = update_data.get("more_details", "")
        else:
            merged["more_details"] = existing_more

        set_payload = {k: merged.get(k) for k in list(allowed_fields) + ["field_validations"]}
        set_payload["edited_at"] = now_ist()

        collection.update_one({"_id": ObjectId(card_id)}, {"$set": set_payload})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": JSONEncoder.encode(updated)}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("update_card error")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_card/{card_id}", status_code=status.HTTP_200_OK)
def delete_card(card_id: str):
    try:
        result = collection.delete_one({"_id": ObjectId(card_id)})
        if result.deleted_count == 1:
            return {"message": "Deleted"}
        else:
            raise HTTPException(status_code=404, detail="Card not found.")
    except Exception as e:
        logging.exception("delete_card error")
        raise HTTPException(status_code=500, detail=str(e))
