# backend.py
import os
import io
import re
from datetime import datetime
from typing import List, Optional, Any, Dict, Tuple

from fastapi import FastAPI, File, UploadFile, Body, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import pytz
import logging

# --- Additional libs for classification/validation ---
import phonenumbers
import tldextract
import validators

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

# -----------------------------------------
# Utilities: OCR parsing + heuristics
# -----------------------------------------
def extract_details(text: str) -> Dict[str, Any]:
    """
    OCR parsing with improved logic to avoid picking company as name.
    Prioritizes ALL-CAPS prominent lines near the top as person names and,
    if company and name collide, searches for alternatives above the company line.
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
        "more_details": "",
        "additional_notes": raw_text,
    }

    # Helper tokens
    company_keywords = [
        "pvt", "private", "ltd", "llp", "inc", "solutions",
        "technologies", "tech", "corporation", "company", "corp", "industries", "works", "enterprises"
    ]
    address_tokens = [
        "street", "st", "road", "rd", "nagar", "lane", "city", "tamilnadu", "india", "pincode",
        "pin", "near", "opp", "zip", "avenue", "av", "bldg", "building", "suite", "ste", "floor",
        "coimbatore", "peelamedu"
    ]

    # EMAIL
    email_m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
    data["email"] = email_m.group(0) if email_m else ""

    # WEBSITE
    website_m = re.search(
        r"(https?://\S+|www\.\S+|\b(?![\w\.-]+@)[A-Za-z0-9\-]+\.(?:com|in|net|org|co|io|biz|info|xyz|me)\b)",
        raw_text,
    )
    data["website"] = website_m.group(0) if website_m else ""

    # PHONES (exclude short numbers like 6-digit pincodes)
    phones = re.findall(r"\+?\d[\d \-\(\)xextEXT]{6,}\d", raw_text)
    normed = []
    for p in phones:
        cleaned = re.sub(r"[^\d\+]", "", p)
        digits_only = re.sub(r"[^\d]", "", cleaned)
        if len(digits_only) >= 8:
            normed.append(cleaned)
    data["phone_numbers"] = list(dict.fromkeys(normed))

    # SOCIAL LINKS / HANDLES
    for l in lines:
        low = l.lower()
        if any(s in low for s in ["linkedin", "instagram", "facebook", "twitter", "x.com", "t.me", "wa.me", "telegram"]):
            data["social_links"].append(l.strip())
        else:
            if re.search(r"[a-z0-9_\-]+-[a-z0-9_\-]+", low) and "@" not in low:
                data["social_links"].append(l.strip())

    # DESIGNATION
    designation_keywords = [
        "founder", "ceo", "cto", "coo", "manager",
        "director", "engineer", "consultant", "head", "lead",
        "president", "vp", "vice", "principal", "officer"
    ]
    for line in lines:
        low = line.lower()
        if any(kw in low for kw in designation_keywords):
            words = line.split()
            limited = " ".join(words[:6])
            clean = re.sub(r"[^A-Za-z&\s\-\./]", " ", limited).strip()
            clean = re.sub(r"\b(?:fm|fin|fmr)\b", "", clean, flags=re.I).strip()
            tokens = [t for t in clean.split() if not re.search(r"-", t)]
            data["designation"] = re.sub(r"\s{2,}", " ", " ".join(tokens)).strip()
            break

    # COMPANY detection: prefer explicit company keywords, otherwise fallback to longer alpha line
    company_candidates = []
    for idx, line in enumerate(lines):
        low = line.lower().strip()
        if re.search(r"[\w\.-]+@[\w\.-]+", line) or re.search(r"\+?\d", line):
            continue
        if any(tok in low for tok in address_tokens):
            continue
        if any(kw in low for kw in company_keywords):
            company_candidates.append((idx, line.strip()))
    if not company_candidates:
        for idx, line in enumerate(lines):
            low = line.lower().strip()
            if re.search(r"[\w\.-]+@[\w\.-]+", line) or re.search(r"\+?\d", line):
                continue
            if any(tok in low for tok in address_tokens):
                continue
            clean_alpha = re.sub(r"[^A-Za-z\s&\.\-]", "", line).strip()
            if not clean_alpha:
                continue
            if 2 <= len(clean_alpha.split()) <= 6 and len(clean_alpha) <= 100:
                if not (clean_alpha.replace(" ", "").isupper() and len(clean_alpha.split()) <= 4):
                    company_candidates.append((idx, clean_alpha))
                    break
    if company_candidates:
        # prefer candidates with explicit company tokens
        company_candidates.sort(key=lambda t: 0 if any(k in t[1].lower() for k in ["private", "pvt", "ltd", "llp", "inc"]) else 1)
        data["company"] = company_candidates[0][1].strip()
        company_idx = company_candidates[0][0]
    else:
        data["company"] = ""
        company_idx = None

    # NAME detection with strong preference rules
    # Top region first (usually prominent branding and name)
    top_region = lines[:6] if len(lines) >= 6 else lines

    def is_person_like(l):
        cleaned = re.sub(r"[^A-Za-z\s]", "", l).strip()
        if not cleaned: 
            return False
        words = cleaned.split()
        return 1 <= len(words) <= 4 and len(cleaned) <= 60

    name_candidate = ""

    # 1) ALL-CAPS lines in top region that are not company/address/phone/email
    for idx, l in enumerate(top_region):
        cleaned = re.sub(r"[^A-Za-z\s]", "", l).strip()
        if cleaned and cleaned.replace(" ", "").isupper() and 1 < len(cleaned.split()) <= 4:
            low = l.lower()
            if not any(kw in low for kw in company_keywords) and not any(tok in low for tok in address_tokens) and "@" not in low and not re.search(r"\+?\d", l):
                name_candidate = cleaned.title()
                break

    # 2) Title-case / capitalized line in top region
    if not name_candidate:
        for idx, l in enumerate(top_region):
            if not is_person_like(l):
                continue
            low = l.lower()
            if any(kw in low for kw in company_keywords) or any(tok in low for tok in address_tokens) or "@" in low or re.search(r"\+?\d", l):
                continue
            words = re.sub(r"[^A-Za-z\s]", "", l).strip().split()
            capitalized = sum(1 for w in words if w[:1].isupper())
            if capitalized >= 1:
                name_candidate = " ".join([w.capitalize() for w in words])
                break

    # 3) spaCy PERSON (if available)
    if not name_candidate and nlp:
        try:
            doc = nlp(" ".join(lines))
            persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if persons:
                for p in persons:
                    p_clean = p.strip()
                    if p_clean and len(p_clean.split()) <= 4:
                        name_candidate = p_clean
                        break
        except Exception:
            pass

    # 4) Conservative scan for any person-like line (fallback)
    if not name_candidate:
        for idx, l in enumerate(lines):
            if re.search(r"[\w\.-]+@[\w\.-]+", l) or re.search(r"\+?\d", l):
                continue
            low = l.lower()
            if any(tok in low for tok in address_tokens) or any(kw in low for kw in company_keywords):
                continue
            cleaned = re.sub(r"[^A-Za-z\s]", "", l).strip()
            if cleaned and 1 <= len(cleaned.split()) <= 4:
                name_candidate = " ".join([w.capitalize() for w in cleaned.split()])
                break

    # If name candidate looks like the company (or equals it), try to find another candidate above company line
    if name_candidate:
        comp = (data.get("company") or "").strip()
        low_name = name_candidate.lower()
        low_comp = comp.lower()
        looks_like_company = any(kw in low_name for kw in company_keywords) or (low_comp and (low_comp in low_name or low_name in low_comp))
        if looks_like_company:
            # search above company_idx if we have it, else search top_region excluding company line
            alt_candidate = ""
            search_limit = company_idx if company_idx is not None else min(len(lines), 6)
            for i in range(0, search_limit):
                l = lines[i]
                if re.search(r"[\w\.-]+@[\w\.-]+", l) or re.search(r"\+?\d", l):
                    continue
                low = l.lower()
                if any(tok in low for tok in address_tokens) or any(kw in low for kw in company_keywords):
                    continue
                if is_person_like(l):
                    cleaned = re.sub(r"[^A-Za-z\s]", "", l).strip()
                    if cleaned and cleaned.replace(" ", "").isupper():
                        alt_candidate = cleaned.title()
                        break
                    words = cleaned.split()
                    capitalized = sum(1 for w in words if w[:1].isupper())
                    if capitalized >= 1:
                        alt_candidate = " ".join([w.capitalize() for w in words])
                        break
            if alt_candidate:
                name_candidate = alt_candidate
            else:
                # final fallback: keep original but strip obvious company tokens
                name_candidate = re.sub(r"\b(private|pvt|ltd|llp|inc|technologies|tech|works|solutions)\b", "", name_candidate, flags=re.I).strip()

    data["name"] = (name_candidate or "").strip()

    # ADDRESS extraction
    address_lines = []
    for l in lines:
        if re.search(r"\b\d{6}\b", l) or re.search(r"\b(?:street|st|road|rd|nagar|lane|peelamedu|city|tamil nadu|coimbatore|near|opp)\b", l, re.I):
            address_lines.append(l)
    if address_lines:
        data["address"] = ", ".join(address_lines)

    # Trim & cleanup
    for k in ["name", "designation", "company", "address", "email", "website", "more_details"]:
        if isinstance(data.get(k), str):
            data[k] = data[k].strip()

    data["name"] = re.sub(r"^[\W_]+|[\W_]+$", "", data.get("name", ""))
    data["company"] = re.sub(r"^[\W_]+|[\W_]+$", "", data.get("company", ""))

    return data


# -----------------------------------------
# Timestamp helper
# -----------------------------------------
def now_ist() -> str:
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------------------
# Classification helpers (unchanged)
# -----------------------------------------
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
    Validate & enrich contact. Keeps field_validations but leaves more_details empty
    unless provided by the caller (frontend controls it).
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

    c["field_validations"] = field_validations

    # Keep more_details empty unless provided
    if "more_details" not in c or not c.get("more_details"):
        c["more_details"] = ""

    # Ensure normalized types
    c["phone_numbers"] = phones
    c["social_links"] = socials

    return c

# -----------------------------------------
# Routes (unchanged)
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

        # classify & enrich before storing (computes validations but leaves more_details empty)
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
                # preserve existing more_details if user didn't provide one
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

        # fetch existing doc
        existing = collection.find_one({"_id": ObjectId(card_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Card not found.")

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
