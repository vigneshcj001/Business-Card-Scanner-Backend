# backend.py
import os
import io
import re
import json
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Body, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image, ImageOps
import pytesseract

# -----------------------
# Requirements:
# pip install fastapi uvicorn python-multipart pillow pytesseract pymongo openai
# Tesseract binary must be installed on host (apt / brew / Windows installer).
# -----------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("business-card-backend")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Optional OpenAI client (only if you install and configure openai)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------
# Helpers & normalization
# -----------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _ensure_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    if isinstance(v, (int, float)):
        return [str(v)]
    if isinstance(v, str):
        items = [x.strip() for x in re.split(r"[,\n;]+", v) if x.strip()]
        return items
    try:
        return [str(v)]
    except Exception:
        return []

def clean_ocr_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def normalize_payload(payload: dict) -> dict:
    out = {}
    out["name"] = payload.get("name")
    out["designation"] = payload.get("designation") or payload.get("title")
    out["company"] = payload.get("company")
    out["phone_numbers"] = _ensure_list(payload.get("phone_numbers") or payload.get("phone") or payload.get("phones"))
    out["email"] = payload.get("email")
    out["website"] = payload.get("website")
    out["address"] = payload.get("address")
    out["social_links"] = _ensure_list(payload.get("social_links") or payload.get("social") or payload.get("linkedin"))
    out["more_details"] = payload.get("more_details") or ""
    out["additional_notes"] = payload.get("additional_notes") or ""
    return out

def db_doc_to_canonical(doc: dict) -> dict:
    if not doc:
        return {}
    canonical = {
        "_id": str(doc.get("_id")),
        "name": doc.get("name"),
        "designation": doc.get("designation"),
        "company": doc.get("company"),
        "phone_numbers": doc.get("phone_numbers") or [],
        "email": doc.get("email"),
        "website": doc.get("website"),
        "address": doc.get("address"),
        "social_links": doc.get("social_links") or [],
        "more_details": doc.get("more_details") or "",
        "additional_notes": doc.get("additional_notes") or "",
        "created_at": doc.get("created_at"),
        "edited_at": doc.get("edited_at"),
        "field_validations": doc.get("field_validations", {}),
    }
    if "raw_text" in doc:
        canonical["raw_text"] = doc.get("raw_text")
    if "confidence_notes" in doc:
        canonical["confidence_notes"] = doc.get("confidence_notes")
    if "extra" in doc:
        canonical["extra"] = doc.get("extra")
    return canonical

# -----------------------
# Local regex-based extractor (fallback / augmentation)
# -----------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-])?(?:\(?\d{2,4}\)?[\s.-])?\d{3,4}[\s.-]?\d{3,4}")
WWW_RE = re.compile(r"(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?", re.I)

COMPANY_HINTS = [
    r"\b(Ltd|Pvt|Private|LLP|Limited|Inc|Corporation|Company|Technologies|Tech|Solutions|Works|Consultants|Advisory|Group|Systems)\b"
]

def local_parse_from_ocr(ocr_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Returns (parsed_dict, confidence_notes)
    parsed_dict keys: name, company, title, email, phone, website, address, extra
    """
    parsed = {
        "name": None,
        "company": None,
        "title": None,
        "email": None,
        "phone": None,
        "website": None,
        "address": None,
        "extra": {},
        "confidence_notes": None,
    }

    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    # Extract emails
    emails = EMAIL_RE.findall(ocr_text)
    if emails:
        parsed["email"] = emails[0].strip()
        parsed["extra"]["emails_all"] = emails

    # Extract websites
    wwws = WWW_RE.findall(ocr_text)
    if wwws:
        wwws_clean = []
        for w in wwws:
            w = w.strip().rstrip(".,;")
            if "@" in w:
                continue
            wwws_clean.append(w)
        if wwws_clean:
            parsed["website"] = wwws_clean[0]
            parsed["extra"]["websites_all"] = wwws_clean

    # Extract phone numbers: regex picks many false positives; filter by length
    phones_raw = PHONE_RE.findall(ocr_text)
    phones = []
    # PHONE_RE.findall returns matches as strings (or tuples depending on groups) — normalize:
    if phones_raw:
        for m in phones_raw:
            if isinstance(m, tuple):
                p = "".join(m)
            else:
                p = m
            p_clean = re.sub(r"[^\d\+]", "", p)
            if 7 <= len(p_clean) <= 15:
                phones.append(p_clean)
    if phones:
        parsed["phone"] = phones[0]
        parsed["extra"]["phones_all"] = phones

    # Guess name:
    name_candidate = None
    for i, ln in enumerate(lines[:6]):  # only look at top few lines
        if len(ln) < 2:
            continue
        if EMAIL_RE.search(ln) or WWW_RE.search(ln):
            continue
        words = ln.split()
        alpha_ratio = sum(c.isalpha() for c in ln) / max(1, len(ln))
        if 1 <= len(words) <= 4 and alpha_ratio > 0.5:
            if not re.search(r"\b(CEO|Founder|Director|Partner|Manager|Consultant|LLP|Pvt|Ltd|Inc|Company|Technologies)\b", ln, re.I):
                name_candidate = ln
                break
    if name_candidate:
        parsed["name"] = name_candidate

    # Guess title: look after name for short lines with caps / known titles
    if parsed["name"]:
        try:
            idx = lines.index(parsed["name"])
            for ln in lines[idx+1: idx+4]:
                if re.search(r"\b(Founder|CEO|Director|Manager|Partner|Consultant|Officer|Advisory|Placement|Head|Chief)\b", ln, re.I):
                    parsed["title"] = ln
                    break
        except ValueError:
            pass

    # Guess company: prefer line containing company hints, else line after name if it looks like company
    company_candidate = None
    for ln in lines:
        for hint_re in COMPANY_HINTS:
            if re.search(hint_re, ln, re.I):
                company_candidate = ln
                break
        if company_candidate:
            break
    if not company_candidate and parsed["name"]:
        try:
            idx = lines.index(parsed["name"])
            if idx+1 < len(lines):
                cand = lines[idx+1]
                if len(cand.split()) <= 6 and any(c.isalpha() for c in cand):
                    if not EMAIL_RE.search(cand) and not PHONE_RE.search(cand):
                        company_candidate = cand
        except ValueError:
            pass
    if company_candidate:
        parsed["company"] = company_candidate

    # Address guess: search for long lines with digits and common address keywords
    addresses = []
    for ln in lines:
        if len(ln) > 30 and re.search(r"\d", ln) and ("," in ln or "Road" in ln or "Street" in ln or "Bengaluru" in ln or "Bangalore" in ln or "Kolhapur" in ln or "Coimbatore" in ln):
            addresses.append(ln)
    if addresses:
        parsed["address"] = " | ".join(addresses[:2])
    else:
        addr_lines = []
        for ln in lines[-6:]:
            if any(keyword in ln.lower() for keyword in ("road", "rd", "street", "st", "bengaluru", "bangalore", "kolhapur", "mumbai", "coimbatore", "address", "city", "block", "floor")) or re.search(r"\d{5,6}", ln):
                addr_lines.append(ln)
        if addr_lines:
            parsed["address"] = ", ".join(addr_lines)

    # If we found nothing for name, last resort: first decent alphabetic line
    if not parsed["name"]:
        for ln in lines[:6]:
            if len(ln.split()) <= 4 and sum(c.isalpha() for c in ln) / max(1, len(ln)) > 0.5:
                parsed["name"] = ln
                break

    # confidence notes summarizing what we found
    notes = []
    if parsed.get("email"):
        notes.append("email_ok")
    if parsed.get("phone"):
        notes.append("phone_ok")
    if parsed.get("website"):
        notes.append("website_ok")
    if parsed.get("name"):
        notes.append("name_guess")
    if parsed.get("company"):
        notes.append("company_guess")
    if parsed.get("address"):
        notes.append("address_guess")
    if not notes:
        notes = ["no_fields_parsed_locally"]

    parsed["confidence_notes"] = ";".join(notes)
    return parsed, parsed["confidence_notes"]

# -----------------------
# OpenAI parsing wrapper
# -----------------------
PARSER_PROMPT = (
    "You are an assistant that extracts structured contact fields from messy OCR'd text from a business card.\n"
    "Return a JSON object with keys: name, company, title, email, phone, website, address, extra, confidence_notes.\n"
    "If a field is not present, set it to null. For 'extra' include any other useful strings (fax, linkedin, notes).\n"
    "Respond ONLY with the JSON object.\n"
)

def call_openai_parse(ocr_text: str, api_key: str, model: str = "gpt-4o") -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed (pip install openai)")
    client = OpenAI(api_key=api_key)
    prompt = PARSER_PROMPT + "\nOCR_TEXT:\n" + ocr_text + "\n\nRespond with JSON only."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a JSON-only extractor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    try:
        assistant_text = resp.choices[0].message.content.strip()
    except Exception:
        assistant_text = str(resp)
    try:
        parsed = json.loads(assistant_text)
        return parsed
    except Exception:
        m = re.search(r"\{[\s\S]*\}$", assistant_text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {
            "name": None,
            "company": None,
            "title": None,
            "email": None,
            "phone": None,
            "website": None,
            "address": None,
            "extra": {"model_output": assistant_text},
            "confidence_notes": "Model output not parseable as JSON. See extra.model_output."
        }

def call_openai_parse_safe(ocr_text: str, api_key: Optional[str], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Try OpenAI if available+key; otherwise return minimal parsed dict.
    """
    if OpenAI is None or not api_key:
        return {
            "name": None,
            "company": None,
            "title": None,
            "email": None,
            "phone": None,
            "website": None,
            "address": None,
            "extra": {"note": "OpenAI not available; local parse used."},
            "confidence_notes": "No OpenAI parsing performed."
        }
    try:
        return call_openai_parse(ocr_text, api_key=api_key, model=model)
    except Exception as e:
        logger.exception("OpenAI parsing failed")
        return {
            "name": None,
            "company": None,
            "title": None,
            "email": None,
            "phone": None,
            "website": None,
            "address": None,
            "extra": {"openai_error": str(e), "traceback": traceback.format_exc()},
            "confidence_notes": f"OpenAI parse failed: {e}"
        }

# -----------------------
# FastAPI + endpoints
# -----------------------
app = FastAPI(title="Business Card OCR Backend (with local parser + MongoDB)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContactBase(BaseModel):
    name: Optional[str] = None
    designation: Optional[str] = None
    company: Optional[str] = None
    phone_numbers: Optional[List[str]] = []
    email: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    social_links: Optional[List[str]] = []
    more_details: Optional[str] = ""
    additional_notes: Optional[str] = ""

class ExtractedContact(ContactBase):
    raw_text: Optional[str] = None
    confidence_notes: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

@app.get("/ping")
async def ping():
    return {"status": "ok", "time": now_iso()}

@app.post("/extract", response_model=ExtractedContact)
async def extract_card(file: UploadFile = File(...), authorization: Optional[str] = Header(None), model: Optional[str] = "gpt-4o"):
    """
    Upload image -> OCR -> local parse -> (optional OpenAI parse) -> merge -> return structured fields.
    """
    api_key = None
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization.split(" ", 1)[1].strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png/etc)")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.exception("Invalid image uploaded")
        return JSONResponse(status_code=400, content={"detail": f"Invalid image: {e}", "traceback": traceback.format_exc()})

    try:
        max_dim = 1800
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)))
    except Exception:
        logger.exception("resize failed; continuing")

    try:
        raw_text = pytesseract.image_to_string(image)
    except Exception as e:
        logger.exception("OCR failed")
        return JSONResponse(status_code=500, content={"detail": f"OCR failure: {e}", "traceback": traceback.format_exc()})

    raw_text_clean = clean_ocr_text(raw_text)

    # local parse (always)
    local_parsed, local_notes = local_parse_from_ocr(raw_text_clean)

    # try OpenAI parse (may return minimal if OpenAI unavailable)
    openai_parsed = call_openai_parse_safe(raw_text_clean, api_key=api_key, model=model)

    # Merge: prefer OpenAI value if present, otherwise local
    merged = {}
    def pick(key):
        v = openai_parsed.get(key)
        if v is not None and v != "":
            return v
        return local_parsed.get(key)

    merged["name"] = pick("name")
    merged["designation"] = pick("title") or pick("designation")
    merged["company"] = pick("company")
    phones = openai_parsed.get("phone") or openai_parsed.get("phone_numbers") or local_parsed.get("phone") or local_parsed.get("phone_numbers")
    merged["phone_numbers"] = _ensure_list(phones)
    merged["email"] = pick("email")
    merged["website"] = pick("website")
    merged["address"] = pick("address")
    social = openai_parsed.get("social_links") or local_parsed.get("social_links") or openai_parsed.get("linkedin") or local_parsed.get("extra", {}).get("linkedin")
    merged["social_links"] = _ensure_list(social)
    merged["more_details"] = openai_parsed.get("more_details") or local_parsed.get("more_details") or ""
    merged["additional_notes"] = openai_parsed.get("additional_notes") or local_parsed.get("additional_notes") or ""
    merged["raw_text"] = raw_text_clean

    cn = []
    if openai_parsed.get("confidence_notes"):
        cn.append(f"openai:{openai_parsed.get('confidence_notes')}")
    if local_parsed.get("confidence_notes"):
        cn.append(f"local:{local_parsed.get('confidence_notes')}")
    merged["confidence_notes"] = ";".join(cn) if cn else "none"

    merged["extra"] = {
        "local": local_parsed.get("extra", {}),
        "openai": openai_parsed.get("extra", {}),
    }

    return JSONResponse(status_code=200, content=merged)

@app.post("/vcard")
async def vcard_endpoint(payload: ContactBase = Body(...)):
    payload_dict = payload.dict()
    phone = payload_dict.get("phone_numbers", [None])[0] if payload_dict.get("phone_numbers") else None
    vcard_data = {
        "name": payload_dict.get("name"),
        "company": payload_dict.get("company"),
        "title": payload_dict.get("designation"),
        "phone": phone,
        "email": payload_dict.get("email"),
        "website": payload_dict.get("website"),
        "address": payload_dict.get("address"),
    }
    vcard = generate_vcard(vcard_data)
    return StreamingResponse(io.BytesIO(vcard.encode("utf-8")),
                             media_type="text/vcard",
                             headers={"Content-Disposition": "attachment; filename=contact.vcf"})

@app.post("/create_card", status_code=status.HTTP_201_CREATED)
async def create_card(payload: ContactBase = Body(...)):
    try:
        doc = normalize_payload(payload.dict())
        doc["created_at"] = now_iso()
        doc["edited_at"] = ""
        doc.setdefault("field_validations", {})

        # dedupe by email
        if doc.get("email"):
            existing = collection.find_one({"email": doc["email"]})
            if existing:
                if not doc.get("more_details"):
                    doc["more_details"] = existing.get("more_details", "")
                doc["edited_at"] = now_iso()
                collection.update_one({"_id": existing["_id"]}, {"$set": doc})
                updated = collection.find_one({"_id": existing["_id"]})
                return {"message": "Updated existing contact", "data": db_doc_to_canonical(updated)}

        result = collection.insert_one(doc)
        inserted = collection.find_one({"_id": result.inserted_id})
        return {"message": "Inserted Successfully", "data": db_doc_to_canonical(inserted)}
    except Exception as e:
        logger.exception("create_card error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all_cards")
def get_all_cards():
    try:
        docs = list(collection.find().sort([("_id", -1)]))
        canonical_list = [db_doc_to_canonical(d) for d in docs]
        return {"data": canonical_list}
    except Exception as e:
        logger.exception("get_all_cards error")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/update_notes/{card_id}")
def update_notes(card_id: str, payload: dict = Body(...)):
    try:
        ts = now_iso()
        update_payload = {
            "additional_notes": payload.get("additional_notes", ""),
            "edited_at": ts
        }
        collection.update_one({"_id": ObjectId(card_id)}, {"$set": update_payload})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": db_doc_to_canonical(updated)}
    except Exception as e:
        logger.exception("update_notes error")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/update_card/{card_id}")
def update_card(card_id: str, payload: dict = Body(...)):
    try:
        allowed_fields = {
            "name", "designation", "company", "phone_numbers",
            "email", "website", "address", "social_links",
            "additional_notes", "more_details"
        }
        normalized = normalize_payload(payload)
        update_data = {k: v for k, v in normalized.items() if k in allowed_fields}
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields to update.")

        existing = collection.find_one({"_id": ObjectId(card_id)})
        if not existing:
            raise HTTPException(status_code=404, detail="Card not found.")

        merged = dict(existing)
        merged.update(update_data)
        merged["edited_at"] = now_iso()
        collection.update_one({"_id": ObjectId(card_id)}, {"$set": merged})
        updated = collection.find_one({"_id": ObjectId(card_id)})
        return {"message": "Updated", "data": db_doc_to_canonical(updated)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("update_card error")
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
        logger.exception("delete_card error")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# vCard helper (placed after endpoints for clarity)
# -----------------------
def generate_vcard(data: Dict[str, Optional[str]]) -> str:
    lines = ["BEGIN:VCARD", "VERSION:3.0"]
    if data.get("name"):
        lines.append(f"FN:{data.get('name')}")
    if data.get("company"):
        lines.append(f"ORG:{data.get('company')}")
    if data.get("title"):
        lines.append(f"TITLE:{data.get('title')}")
    if data.get("phone"):
        lines.append(f"TEL;TYPE=WORK,VOICE:{data.get('phone')}")
    if data.get("email"):
        lines.append(f"EMAIL;TYPE=WORK:{data.get('email')}")
    if data.get("website"):
        lines.append(f"URL:{data.get('website')}")
    if data.get("address"):
        lines.append(f"ADR;TYPE=WORK:;;{data.get('address')}")
    lines.append("END:VCARD")
    return "\n".join(lines)
