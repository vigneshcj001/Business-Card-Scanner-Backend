# backend.py
import os
import io
import re
from datetime import datetime
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, File, UploadFile, Body, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import pytz

# --- Additional libs for classification/validation ---
import phonenumbers
import tldextract
import validators
import logging

# Try loading spaCy NER (optional)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# Load .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")
PHONE_DEFAULT_REGION = os.getenv("PHONE_DEFAULT_REGION", "IN")  # default phone region

# FastAPI setup
app = FastAPI(title="Business Card OCR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to allowed origins in production
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
    email: Optional[EmailStr] = ""
    website: Optional[str] = ""
    address: Optional[str] = ""
    social_links: Optional[List[str]] = []
    more_details: Optional[str] = ""            # added so frontend can send it
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

# -----------------------------------------
# Utilities: improved extract_details (OCR parsing)
# -----------------------------------------
def extract_details(text: str) -> Dict[str, Any]:
    """
    Parse OCR text into structured contact fields with improved name/company heuristics.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    raw_text = " ".join(lines)

    data = {
        "name": "",
        "designation": "",
        "company": "",
        "phone_numbers": [],
        "email": "",
        "website": "",
        "address": "",
        "social_links": [],
        "more_details": "",            # start empty so user fills it
        "additional_notes": raw_text,
    }

    # EMAIL
    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
    data["email"] = email.group(0) if email else ""

    # WEBSITE (prefer explicit http/www)
    website = re.search(r"(https?://\S+|www\.\S+|\S+\.\w{2,})", raw_text)
    data["website"] = website.group(0) if website else ""

    # PHONE NUMBERS (loose capture)
    phones = re.findall(r"\+?\d[\d \-\(\)xextEXT]{6,}\d", raw_text)
    phones = [re.sub(r"[^\d\+xX]", "", p) for p in phones]
    data["phone_numbers"] = list(dict.fromkeys(phones))  # dedupe preserving order

    # SOCIAL LINKS
    for l in lines:
        low = l.lower()
        if any(s in low for s in ["linkedin", "instagram", "facebook", "twitter", "x.com", "t.me", "wa.me", "telegram"]):
            data["social_links"].append(l.strip())

    # -----------------------------------------
    # IMPROVED DESIGNATION LOGIC
    # -----------------------------------------
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead",
        "president", "vp", "vice", "principal", "officer"
    ]

    for line in lines:
        low = line.lower()
        if any(kw in low for kw in designation_keywords):
            words = line.split()
            limited = " ".join(words[:6])  # a few more words allowed
            clean = re.sub(r"[^A-Za-z&\s\-\./]", "", limited).strip()
            # remove small OCR garbage tokens
            clean = re.sub(r"\b(fm|fin|fmr)\b", "", clean, flags=re.I).strip()
            data["designation"] = clean
            break

    # -----------------------------------------
    # IMPROVED COMPANY DETECTION (handles lowercase & avoids addresses)
    # -----------------------------------------
    company_keywords = [
        "pvt", "private", "ltd", "llp", "inc", "solutions",
        "technologies", "tech", "corporation", "company", "corp", "industries", "works", "enterprises"
    ]

    # tokens that strongly suggest an address (skip these when picking company)
    address_tokens = [
        "street", "st", "road", "rd", "nagar", "lane", "city", "tamilnadu", "india", "pincode",
        "pin", "near", "opp", "zip", "avenue", "av", "bldg", "building", "suite", "ste", "floor"
    ]

    def looks_like_name(line: str) -> bool:
        clean = re.sub(r"[^A-Za-z\s]", "", line).strip()
        if not clean:
            return False
        words = clean.split()
        if not (1 <= len(words) <= 4):
            return False
        if len(clean) > 40:
            return False
        # reject if contains common company keywords
        if any(w in line.lower() for w in company_keywords):
            return False
        if re.search(r"\d", line):
            return False
        uppercase_count = sum(1 for w in words if w[:1].isupper())
        if uppercase_count >= 1:
            return True
        if line.islower() and len(words) <= 3:
            return True
        return False

    company_candidates = []
    for line in lines:
        low = line.lower().strip()

        # skip lines with digits (likely an address / phone / street number)
        if re.search(r"\d", low):
            continue

        # skip if line contains address-like tokens
        if any(tok in low for tok in address_tokens):
            continue

        # prefer explicit company keywords
        if any(kw in low for kw in company_keywords):
            company_candidates.append(line.strip())

    # fallback: pick a short non-name, non-address line (more conservative)
    if not company_candidates:
        for line in lines:
            low = line.lower().strip()
            # skip lines that look like email/phones or contain digits
            if re.search(r"[\w\.-]+@[\w\.-]+", line) or re.search(r"\+?\d", line):
                continue
            if any(tok in low for tok in address_tokens):
                continue
            clean_alpha = re.sub(r"[^A-Za-z\s&\.\-]", "", line).strip()
            if not clean_alpha:
                continue
            if 1 <= len(clean_alpha.split()) <= 6 and len(clean_alpha) <= 60:
                # avoid picking pure-person-looking lines (let name logic handle that)
                if not looks_like_name(line):
                    company_candidates.append(line.strip())
                    break

    if company_candidates:
        data["company"] = company_candidates[0]

    # -----------------------------------------
    # IMPROVED NAME DETECTION (lowercase & mixed case)
    # -----------------------------------------
    name_candidates = []
    for line in lines:
        # strip leading noise like copyright/trademark bullets/symbols
        clean_line = re.sub(r"^[\u00A9\u00AE©®\s\W]+", "", line).strip()

        # ignore obvious contact lines
        if re.search(r"[\w\.-]+@[\w\.-]+", clean_line) or re.search(r"\+?\d", clean_line):
            continue

        # heuristic: looks like a person name
        if looks_like_name(clean_line):
            if clean_line == data.get("company") or clean_line == data.get("designation"):
                continue
            name_candidates.append(clean_line)

    # If spaCy NER available, prefer PERSON entity
    if nlp:
        try:
            full_text = " ".join(lines)
            doc = nlp(full_text)
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if persons:
                for p in persons:
                    if p.strip() and p.strip() != data.get("company"):
                        data["name"] = p.strip()
                        break
        except Exception:
            pass

    # fallback logic if spaCy not available or didn't find anything
    if not data["name"] and name_candidates:
        data["name"] = name_candidates[0]
    elif not data["name"]:
        # previous uppercase heuristic as last resort
        uppercase_lines = []
        for l in lines:
            cleaned = re.sub(r"[^A-Za-z\s]", "", l).strip()
            if cleaned and cleaned.replace(" ", "").isupper() and len(cleaned.split()) <= 4:
                uppercase_lines.append(cleaned)
        if uppercase_lines:
            data["name"] = uppercase_lines[0]
        else:
            # final fallback: first short non-contact, non-company line
            for l in lines:
                if l == data["company"] or l == data["designation"]:
                    continue
                if re.search(r"[\w\.-]+@[\w\.-]+", l):
                    continue
                if re.search(r"\+?\d", l):
                    continue
                if 1 <= len(l.split()) <= 4 and len(l) < 60:
                    cleaned = re.sub(r"[^A-Za-z\s]", "", l).strip()
                    data["name"] = cleaned
                    break

    # ADDRESS heuristics (conservative)
    address_lines = []
    for l in lines:
        if re.search(r"\d.*(street|st|road|rd|nagar|lane|city|tamilnadu|india|pincode|pin|near|opp|zip|avenue|av|bldg|building|suite|ste|floor)", l, re.I):
            address_lines.append(l)
    if address_lines:
        data["address"] = ", ".join(address_lines)

    # Trim strings and final cleanup
    for k in ["name", "designation", "company", "address", "email", "website", "more_details"]:
        if isinstance(data.get(k), str):
            data[k] = data[k].strip()

    # small extra cleanup: remove stray leading/trailing punctuation in name/company
    data["name"] = re.sub(r"^[\W_]+|[\W_]+$", "", data.get("name", ""))
    data["company"] = re.sub(r"^[\W_]+|[\W_]+$", "", data.get("company", ""))

    return data

# -----------------------------------------
# Helper: timestamp
# -----------------------------------------
def now_ist() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------------------
# Classification helper (validate & enrich)
# -----------------------------------------
# NOTE: we compute the validations internally but we do not force/save a populated
# "more_details" here. Instead, new inserts will have more_details="" by default
# (handled in extract_details / create/upload logic). field_validations is kept
# but the frontend hides it from users.
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
        candidate = re.sub(r"[^\d\+xX]", "", raw)
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

def classify_contact(contact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate & enrich contact. Returns a new dict with added:
      - field_validations (structured)
      - more_details is intentionally left/kept empty for user to fill
    """
    c = dict(contact)  # shallow copy

    # Normalize types
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

    # NER quick check using spaCy (optional)
    ner_org = ""
    ner_gpe = ""
    if nlp and (company or address or name):
        try:
            txt = " ".join([x for x in [name, company, address] if x])
            doc = nlp(txt)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
            gpes = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
            ner_org = orgs[0] if orgs else ""
            ner_gpe = gpes[0] if gpes else ""
        except Exception:
            ner_org = ""
            ner_gpe = ""

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
        "ner_org": ner_org,
        "ner_gpe": ner_gpe,
        "notes": {"length": notes_len, "lines": lines_in_notes},
        "designation": {"value": designation}
    }

    # Attach validations, but keep more_details intentionally empty here:
    c["field_validations"] = field_validations
    # DO NOT auto-populate "more_details" — allow user to fill in frontend.
    if "more_details" not in c or not c.get("more_details"):
        c["more_details"] = ""

    # Ensure the phone and social fields stay normalized types
    c["phone_numbers"] = phones
    c["social_links"] = socials

    return c

# -----------------------------------------
# Routes
# -----------------------------------------
@app.get("/")
def root():
    return {"message": "OCR Backend Running ✅"}

@app.post("/upload_card", status_code=status.HTTP_201_CREATED)
async def upload_card(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(img)
        extracted = extract_details(text)

        # classify & enrich before storing
        extracted = classify_contact(extracted)

        # ensure more_details is empty for newly created records (user will fill later)
        extracted["more_details"] = ""

        extracted["created_at"] = now_ist()
        extracted["edited_at"] = ""

        result = collection.insert_one(extracted)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": JSONEncoder.encode(inserted)}
    except Exception as e:
        logging.exception("upload_card error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_card", status_code=status.HTTP_201_CREATED)
async def create_card(payload: ContactCreate):
    try:
        doc = payload.dict()

        # classify & enrich before insert/update
        doc = classify_contact(doc)

        # If user provided more_details in payload, keep it (manual create)
        if payload.more_details:
            doc["more_details"] = payload.more_details
        else:
            doc["more_details"] = ""

        doc["created_at"] = now_ist()
        doc["edited_at"] = ""

        # If email exists, update that doc instead of creating a duplicate
        if doc.get("email"):
            existing = collection.find_one({"email": doc["email"]})
            if existing:
                # merge/overwrite fields and set edited_at
                doc["edited_at"] = now_ist()
                # preserve existing more_details if user didn't provide one
                if not doc.get("more_details"):
                    doc["more_details"] = existing.get("more_details", "")
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

        # fetch existing doc
        existing = collection.find_one({"_id": ObjectId(card_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Card not found.")

        # preserve existing more_details unless user explicitly provided one
        existing_more = existing.get("more_details", "")

        merged = dict(existing)
        merged.update(update_data)

        # classify -> returns normalized phone/socials and adds field_validations
        merged = classify_contact(merged)

        # ensure more_details: if user provided it in update_data, keep that; otherwise preserve existing
        if "more_details" in update_data:
            merged["more_details"] = update_data.get("more_details", "")
        else:
            merged["more_details"] = existing_more

        # pick only allowed fields + classification fields to set
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
