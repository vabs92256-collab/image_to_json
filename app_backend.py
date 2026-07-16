"""
OCR Backend — extracts text from any image and returns structured JSON key-value pairs.

Pipeline:
  1. Accept any image type (jpg, png, bmp, tiff, webp, gif, etc.) via FastAPI upload
  2. Normalize with Pillow -> RGB
  3. Run Tesseract OCR to get raw text
  4. Send raw text to local Ollama (Llama 3.2) to structure into JSON key-value pairs
  5. Return clean JSON response

Run:
    pip install fastapi uvicorn pillow pytesseract requests python-multipart
    # also needs system tesseract: sudo apt-get install tesseract-ocr
    uvicorn ocr_backend:app --reload --port 8000

 omniocr 
 paddleocr + vl 

"""

import io
import json
import logging
import pytesseract
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TEMPERATURE = 0.7
OLLAMA_TIMEOUT = 60

LOG_FILE = "ocr_backend.log"

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp",
    "image/tiff", "image/webp", "image/gif", "image/x-ms-bmp",
}
MAX_FILE_SIZE_MB = 15

STRUCTURE_PROMPT = """You are a document data extraction engine.
Below is raw OCR text extracted from an image. Convert it into a clean,
flat JSON object of key-value pairs representing the meaningful fields
in the document (e.g. names, dates, amounts, IDs, labels found in the text).

Rules:
- Return ONLY valid JSON, no markdown, no commentary.
- Use snake_case keys.
- If a value spans multiple lines/fragments, merge sensibly.
- If nothing structured can be found, return {{"raw_text": "<full text>"}}.

OCR TEXT:
---
{ocr_text}
---

JSON:"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ocr_backend")

app = FastAPI(title="OCR to JSON Backend")


# ---------------------------------------------------------------------------
# CORE STEPS
# ---------------------------------------------------------------------------
def extract_text_from_image(image_bytes: bytes) -> str:
    """Open any supported image format and run OCR on it."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unreadable image file: {e}")

    try:
        text = pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR engine failed: {e}")

    return text.strip()


def structure_text_with_llm(ocr_text: str) -> dict:
    """Send raw OCR text to Ollama and get back structured JSON key-value pairs."""
    if not ocr_text:
        return {"raw_text": ""}

    logger.info("RAW OCR TEXT:\n%s", ocr_text)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": STRUCTURE_PROMPT.format(ocr_text=ocr_text),
        "format": "json",
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Ollama call failed, falling back to raw text: %s", e)
        return {"raw_text": ocr_text}

    result = response.json()
    raw_output = result.get("response", "").strip()

    logger.info("RAW LLM RESPONSE:\n%s", raw_output)

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            return parsed
        return {"raw_text": ocr_text, "model_output": parsed}
    except json.JSONDecodeError:
        logger.warning("LLM did not return valid JSON, falling back to raw text")
        return {"raw_text": ocr_text}


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}",
        )

    image_bytes = await file.read()

    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail="File too large.")

    logger.info("Processing file: %s", file.filename)
    ocr_text = extract_text_from_image(image_bytes)
    structured_json = structure_text_with_llm(ocr_text)

    return JSONResponse(content={
        "filename": file.filename,
        "status": "success",
        "data": structured_json,
    })


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)