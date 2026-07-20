import io
import json
import requests
import pandas as pd
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

engine = RapidOCR()

st.set_page_config(
    page_title="RapidOCR + LLM",
    layout="wide"
)

st.title("📄 OCR → JSON using RapidOCR + Llama")

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"]
)


def pdf_page_to_image(pdf_bytes, page_index, zoom=2.0):
    """Render a single PDF page to a PIL Image + PNG bytes."""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page = doc.load_page(page_index)

    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat)

    png_bytes = pix.tobytes("png")

    image = Image.open(io.BytesIO(png_bytes))

    doc.close()

    return image, png_bytes


if uploaded_file is not None:

    is_pdf = uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf")

    image_bytes = None
    image = None

    if is_pdf:

        pdf_bytes = uploaded_file.getvalue()

        try:

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            num_pages = doc.page_count

            doc.close()

        except Exception as e:

            st.error(e)
            st.stop()

        if num_pages > 1:

            page_number = st.selectbox(
                "Select PDF page",
                options=list(range(1, num_pages + 1)),
                index=0
            )

        else:

            page_number = 1

        try:

            image, image_bytes = pdf_page_to_image(
                pdf_bytes,
                page_number - 1
            )

        except Exception as e:

            st.error(e)
            st.stop()

    else:

        image_bytes = uploaded_file.getvalue()

        image = Image.open(io.BytesIO(image_bytes))

    left, right = st.columns([1, 1.4])

    # ----------------------------------------------------
    # LEFT
    # ----------------------------------------------------
    with left:

        st.subheader("Uploaded Image" if not is_pdf else "PDF Page Preview")

        st.image(
            image,
            width="stretch"
        )

    # ----------------------------------------------------
    # RIGHT
    # ----------------------------------------------------
    with right:

        with st.spinner("Running OCR..."):

            try:

                result, _ = engine(image_bytes)

            except Exception as e:

                st.error(e)
                st.stop()

        raw_text = ""

        if result:

            raw_text = "\n".join(
                [line[1] for line in result]
            )

        st.subheader("Raw OCR Output")

        st.text_area(
            "Raw OCR",
            raw_text,
            height=250,
            label_visibility="collapsed"
        )

        if raw_text.strip() == "":

            st.warning("No text detected.")
            st.stop()

        prompt = f"""
You are an intelligent document extraction system.

OCR TEXT
---------------------
{raw_text}
---------------------

Extract every possible field.

Return ONLY valid JSON.

Required JSON format:

{{
    "document_type": "",

    "fields": {{

    }},

    "tables": [

        {{
            "table_name": "",

            "rows": [
                {{}}
            ]
        }}

    ],

    "summary": ""
}}

Rules:

1. Return ONLY valid JSON matching the structure above. No markdown, no commentary, no code fences.

2. Populate "fields" with every key-value pair detected in the document (e.g. Name, Address, Date, Invoice Number, GST, IFSC, MICR, Account Number, Cheque Number, Amount, Phone, Email).

3. If the document contains one or more tables:
   - Extract each as a separate object inside "tables"
   - Each row must be a flat JSON object with column names as keys
   - Preserve row order as it appears in the document

4. If no table is present, return "tables": [] (empty array, not null or omitted).

5. Do not invent or infer values that are not present in the OCR text.

"""

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        with st.spinner("Generating JSON..."):

            try:

                response = requests.post(
                    OLLAMA_URL,
                    json=payload,
                    timeout=300
                )

                response.raise_for_status()

                response_json = response.json()

                llm_output = response_json.get(
                    "response",
                    ""
                )

            except Exception as e:

                st.error(e)
                st.stop()

        # ----------------------------------------------------
        # RAW LLM OUTPUT
        # ----------------------------------------------------

        st.subheader("LLM Raw Response")

        st.code(
            llm_output,
            language="json"
        )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        st.subheader("Structured JSON")

        try:

            parsed = json.loads(llm_output)

            st.json(parsed)

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            summary = parsed.get(
                "summary",
                ""
            )

            if summary:

                st.subheader("Summary")

                st.info(summary)

            # ------------------------------------------------
            # TABLES
            # ------------------------------------------------

            tables = parsed.get(
                "tables",
                []
            )

            if tables:

                st.subheader("Detected Tables")

                for idx, table in enumerate(tables):

                    st.markdown(
                        f"### {table.get('table_name', f'Table {idx+1}')}"
                    )

                    rows = table.get(
                        "rows",
                        []
                    )

                    if rows:

                        df = pd.DataFrame(rows)

                        st.dataframe(
                            df,
                            width="stretch",
                            hide_index=True
                        )

            # ------------------------------------------------
            # FIELDS
            # ------------------------------------------------

            fields = parsed.get(
                "fields",
                {}
            )

            if fields:

                st.subheader("Extracted Fields")

                df = pd.DataFrame(
                    list(fields.items()),
                    columns=["Field", "Value"]
                )

                st.dataframe(
                    df,
                    width="stretch",
                    hide_index=True
                )

        except Exception:

            st.error("Model did not return valid JSON.")

            st.code(llm_output)