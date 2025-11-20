# backend.py
import os
import io
import re
import json
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Body, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from PIL import Image
import pytesseract

# -----------------------
# Logging & config
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "business_cards")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contacts")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Optional OpenAI client (only used for parsing OCR output)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------
# Helpers & normalization
# -----------------------
def now_ist() -> str:
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
# Pydantic models (canonical)
# -----------------------
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

# -----------------------
# Parser / vCard helpers
# -----------------------
PARSER_PROMPT = (
    "You are an assistant that extracts structured contact fields from messy OCR'd text from a business card.\n"
    "Return a JSON object with keys: name, company, title, email, phone, website, address, extra.\n"
    "If a field is not present, set it to null. For 'extra' include any other useful strings (fax, linkedin, notes).\n"
    "Also add a short field 'confidence_notes' describing any ambiguity.\n\n"
    "Respond ONLY with the JSON object.\n"
)

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

def call_openai_parse_safe(ocr_text: str, api_key: Optional[str], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Try calling OpenAI if available and api_key provided.
    If OpenAI not installed or no key, return a safe minimal parsed dict (OCR-only).
    Any exceptions from OpenAI are caught and returned as 'extra' and 'confidence_notes'.
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
            "extra": {"note": "OpenAI not available; returning OCR text only."},
            "confidence_notes": "No OpenAI parsing performed."
        }

    try:
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
    except Exception as e:
        logger.exception("OpenAI call failed")
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
# FastAPI app
# -----------------------
app = FastAPI(title="Business Card OCR Backend (Full)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/extract", response_model=ExtractedContact)
async def extract_card(file: UploadFile = File(...), authorization: Optional[str] = Header(None), model: Optional[str] = "gpt-4o"):
    """
    Dev-friendly /extract:
     - Logs exceptions to console.
     - If no OpenAI key or SDK available, falls back to OCR-only result instead of 500.
     - Includes helpful error details in response (dev).
    """
    api_key = None
    if authorization and authorization.lower().startswith("bearer "):
        api_key = authorization.split(" ", 1)[1].strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

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
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)
    except Exception:
        logger.exception("Image resize failed — continuing with original image")

    try:
        raw_text = pytesseract.image_to_string(image)
    except Exception as e:
        logger.exception("OCR failed")
        return JSONResponse(status_code=500, content={"detail": f"OCR failure: {e}", "traceback": traceback.format_exc()})

    raw_text = clean_ocr_text(raw_text)

    parsed = call_openai_parse_safe(raw_text, api_key=api_key, model=model)

    try:
        result = {
            "name": parsed.get("name"),
            "designation": parsed.get("title") or parsed.get("designation"),
            "company": parsed.get("company"),
            "phone_numbers": _ensure_list(parsed.get("phone") or parsed.get("phone_numbers")),
            "email": parsed.get("email"),
            "website": parsed.get("website"),
            "address": parsed.get("address"),
            "social_links": _ensure_list(parsed.get("social_links") or parsed.get("linkedin") or parsed.get("social")),
            "more_details": parsed.get("more_details") or "",
            "additional_notes": parsed.get("additional_notes") or "",
            "raw_text": raw_text,
            "confidence_notes": parsed.get("confidence_notes"),
            "extra": parsed.get("extra"),
        }
    except Exception as e:
        logger.exception("Normalization failed")
        return JSONResponse(status_code=500, content={"detail": f"Normalization failed: {e}", "traceback": traceback.format_exc(), "parsed": parsed})

    return JSONResponse(status_code=200, content=result)

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
        doc["created_at"] = now_ist()
        doc["edited_at"] = ""
        doc.setdefault("field_validations", {})

        if doc.get("email"):
            existing = collection.find_one({"email": doc["email"]})
            if existing:
                if not doc.get("more_details"):
                    doc["more_details"] = existing.get("more_details", "")
                doc["edited_at"] = now_ist()
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
        ts = now_ist()
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
        merged["edited_at"] = now_ist()
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
