"""
OCR Backend — extracts text from any image OR PDF and returns structured JSON key-value pairs.

Pipeline:
  1. Accept any image type (jpg, png, bmp, tiff, webp, gif, etc.) OR a PDF via FastAPI upload
  2a. Images: normalize with Pillow -> RGB -> Tesseract OCR
  2b. PDFs: open with PyMuPDF, try native text layer per page; if a page has no
      extractable text (i.e. it's a scanned/image-only page), rasterize that page
      and run it through the same Tesseract OCR path
  3. Send raw text to local Ollama (Llama 3.2) to structure into JSON key-value pairs
  4. Return clean JSON response

Run:
    pip install fastapi uvicorn pillow pytesseract requests python-multipart pymupdf
    # also needs system tesseract: sudo apt-get install tesseract-ocr
    uvicorn ocr_backend:app --reload --port 8000
"""

import io
import json
import logging
import pytesseract
import requests
import fitz  # PyMuPDF
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

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/bmp",
    "image/tiff", "image/webp", "image/gif", "image/x-ms-bmp",
}
ALLOWED_PDF_CONTENT_TYPES = {"application/pdf"}
ALLOWED_CONTENT_TYPES = ALLOWED_IMAGE_CONTENT_TYPES | ALLOWED_PDF_CONTENT_TYPES

MAX_FILE_SIZE_MB = 15
MAX_PDF_PAGES = 25  # guardrail so a huge PDF doesn't hang the request
PDF_RENDER_DPI = 300  # used only for pages that need OCR fallback
MIN_CHARS_FOR_NATIVE_TEXT = 20  # below this, treat page as scanned/image-only

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
# CORE STEPS — IMAGES
# ---------------------------------------------------------------------------
def ocr_pil_image(image: Image.Image) -> str:
    """Run Tesseract on a single PIL image, RGB-normalized."""
    try:
        image = image.convert("RGB")
        text = pytesseract.image_to_string(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR engine failed: {e}")
    return text.strip()


def extract_text_from_image(image_bytes: bytes) -> str:
    """Open any supported image format and run OCR on it."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unreadable image file: {e}")

    return ocr_pil_image(image)


# ---------------------------------------------------------------------------
# CORE STEPS — PDFs
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Walk every page of the PDF:
      - if the page has a native text layer, pull it directly (fast, exact)
      - if not (scanned/image-only page), rasterize the page and OCR it
    Page texts are concatenated with page-break markers so downstream
    structuring still sees the whole document as one blob of raw text.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unreadable PDF file: {e}")

    if doc.page_count == 0:
        raise HTTPException(status_code=400, detail="PDF has no pages.")

    if doc.page_count > MAX_PDF_PAGES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF has {doc.page_count} pages; limit is {MAX_PDF_PAGES}.",
        )

    page_texts = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        native_text = page.get_text().strip()

        if len(native_text) >= MIN_CHARS_FOR_NATIVE_TEXT:
            logger.info("Page %d: using native text layer", page_index + 1)
            page_texts.append(native_text)
            continue

        logger.info("Page %d: no usable text layer, falling back to OCR", page_index + 1)
        try:
            pixmap = page.get_pixmap(dpi=PDF_RENDER_DPI)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            ocr_text = ocr_pil_image(image)
            page_texts.append(ocr_text)
        except Exception as e:
            logger.warning("Page %d OCR fallback failed: %s", page_index + 1, e)
            page_texts.append("")

    doc.close()

    combined = "\n\n--- page break ---\n\n".join(t for t in page_texts if t)
    return combined.strip()


# ---------------------------------------------------------------------------
# CORE STEPS — LLM STRUCTURING (unchanged, format-agnostic)
# ---------------------------------------------------------------------------
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

    file_bytes = await file.read()

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail="File too large.")

    logger.info("Processing file: %s (%s)", file.filename, file.content_type)

    if file.content_type in ALLOWED_PDF_CONTENT_TYPES:
        ocr_text = extract_text_from_pdf(file_bytes)
    else:
        ocr_text = extract_text_from_image(file_bytes)

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