"""
OCR Backend — extracts text from any image OR PDF and returns structured JSON key-value pairs.

Pipeline:
  1. Accept any image type (jpg, png, bmp, tiff, webp, gif, etc.) OR a PDF via FastAPI upload
  2a. Images: normalize with Pillow -> RGB -> PaddleOCR
  2b. PDFs: open with PyMuPDF, try native text layer per page; if a page has no
      extractable text (i.e. it's a scanned/image-only page), rasterize that page
      and run it through the same PaddleOCR path
  3. Send raw text to local Ollama (Llama 3.2) to structure into JSON key-value pairs
  4. Return clean JSON response

Run:
    pip install fastapi uvicorn pillow requests python-multipart pymupdf numpy
    pip install paddlepaddle paddleocr
    # PaddleOCR downloads its detection/recognition/(angle-classifier) models on
    # first run and caches them locally — no system package needed (unlike Tesseract).
    uvicorn ocr_backend:app --reload --port 8000

Note on PaddleOCR versions:
    This targets the PaddleOCR 3.x API (`PaddleOCR(use_textline_orientation=..., lang=..., device=...)`,
    `.ocr(img)` with no per-call kwargs, and a dict-like `OCRResult` with a
    `rec_texts` field). If you're on 2.x instead, the constructor takes
    `use_angle_cls`/`use_gpu`/`show_log` and `.ocr()` takes `cls=True` — the
    result parser below (`_lines_from_paddle_result`) already handles both
    result shapes, but PADDLE_OCR_INIT_KWARGS would need to switch to the
    2.x names.
"""

import io
import json
import logging
import numpy as np
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from paddleocr import PaddleOCR

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

PADDLE_OCR_INIT_KWARGS = {
    "use_textline_orientation": True,  # 3.x name for the old use_angle_cls
    "lang": "en",
    "device": "cpu",  # 3.x name for the old use_gpu; use "gpu" or e.g. "gpu:0" for CUDA
}

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
paddle_ocr = PaddleOCR(**PADDLE_OCR_INIT_KWARGS)


# ---------------------------------------------------------------------------
# CORE STEPS — IMAGES
# ---------------------------------------------------------------------------
def ocr_pil_image(image: Image.Image) -> str:
    """Run PaddleOCR on a single PIL image, RGB-normalized."""
    try:
        image = image.convert("RGB")
        img_array = np.array(image)
        # NOTE: PaddleOCR 3.x dropped the per-call `cls=` kwarg — angle
        # classification is now controlled only via use_textline_orientation
        # (or the legacy use_angle_cls) at construction time. Calling .ocr()
        # with no extra kwargs works on both 2.x and 3.x installs.
        result = paddle_ocr.ocr(img_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR engine failed: {e}")

    return _lines_from_paddle_result(result)


def _lines_from_paddle_result(result) -> str:
    """
    Handles both PaddleOCR result shapes:
      - 2.x: list with one entry per image; each entry is a list of
        [box, (text, confidence)] lines
      - 3.x: list with one entry per image; each entry is a dict-like
        OCRResult with a 'rec_texts' list of recognized strings
    """
    if not result:
        return ""

    page = result[0]
    if not page:
        return ""

    # 3.x dict-like result (also covers OCRResult objects, which subclass dict)
    if isinstance(page, dict) and "rec_texts" in page:
        texts = page.get("rec_texts") or []
        return "\n".join(t for t in texts if t).strip()

    # 3.x result object exposing rec_texts as an attribute instead of a key
    if hasattr(page, "rec_texts"):
        texts = getattr(page, "rec_texts") or []
        return "\n".join(t for t in texts if t).strip()

    # 2.x list-of-lines result
    if isinstance(page, list):
        lines = []
        for line in page:
            if not line or len(line) < 2:
                continue
            text_conf = line[1]
            if not text_conf:
                continue
            text = text_conf[0]
            if text:
                lines.append(text)
        return "\n".join(lines).strip()

    return ""


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