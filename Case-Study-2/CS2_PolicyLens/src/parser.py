"""
Stage 1: Parse (src/parser.py)
Extracts text, section headings, and structured tables per page from PDFs.
- Uses PyMuPDF (pymupdf) for layout-aware block extraction (handling two-column layouts by sorting columns)
- Uses pdfplumber to detect and format bordered tables as intact Markdown tables ([TABLE_START]...[TABLE_END])
- Automatically detects scanned pages (< 50 alphanumeric characters) and falls back to Tesseract OCR at 200 DPI
"""

import io
import re
from typing import Any, Dict, List

import pymupdf as fitz
import pdfplumber
import pytesseract
from PIL import Image

HEADING_REGEX = re.compile(
    r"^(?:Section|Clause|Chapter|Table)\s+\d+(?:\.\d+)?[:\s].*|"
    r"^\d+\.\d+\s+[A-Z][A-Za-z0-9\s,\-\(\)/]+$",
    re.MULTILINE,
)


def _format_table_as_markdown(table: List[List[Any]]) -> str:
    """Converts a 2D list extracted by pdfplumber into a clean Markdown pipe table."""
    if not table or len(table) < 2:
        return ""
    cleaned_rows = []
    for row in table:
        if not row:
            continue
        cells = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        if any(cells):
            cleaned_rows.append(cells)

    if len(cleaned_rows) < 2:
        return ""

    header = cleaned_rows[0]
    sep = ["---" for _ in header]
    md_lines = [
        "[TABLE_START]",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in cleaned_rows[1:]:
        # Pad or trim row to match header length
        padded = (row + [""] * len(header))[: len(header)]
        md_lines.append("| " + " | ".join(padded) + " |")
    md_lines.append("[TABLE_END]")
    return "\n".join(md_lines)


def _extract_column_sorted_text(fitz_page: fitz.Page, midpoint_x: float = 297.5) -> str:
    """
    Extracts text blocks from a PyMuPDF page.
    If blocks sit cleanly in left and right halves (two-column layout), orders left column
    top-to-bottom before right column top-to-bottom so sentences never interleave across columns.
    """
    # Block layout: (x0, y0, x1, y1, text, block_no, block_type).
    blocks = fitz_page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and str(b[4]).strip()]
    if not text_blocks:
        return ""

    # Separate header/footer/full-width blocks from body column blocks
    top_blocks = []
    left_col = []
    right_col = []
    bottom_blocks = []

    page_h = fitz_page.rect.height
    for b in text_blocks:
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], str(b[4]).strip()
        width = x1 - x0
        if y0 < 45:
            top_blocks.append((y0, x0, txt))
        elif y1 > page_h - 45:
            bottom_blocks.append((y0, x0, txt))
        elif width < 260 and x1 <= midpoint_x + 15:
            left_col.append((y0, x0, txt))
        elif width < 260 and x0 >= midpoint_x - 15:
            right_col.append((y0, x0, txt))
        else:
            # Full-width block in upper/middle body
            left_col.append((y0, x0, txt))

    top_blocks.sort()
    left_col.sort()
    right_col.sort()
    bottom_blocks.sort()

    ordered = [t[2] for t in (top_blocks + left_col + right_col + bottom_blocks)]
    return "\n\n".join(ordered)


def _extract_headings_from_text(text: str) -> List[str]:
    """Finds section/clause headings in extracted page text."""
    headings = []
    for line in text.splitlines():
        s = line.strip()
        if HEADING_REGEX.match(s) and len(s) < 140:
            headings.append(s)
    return headings


def parse_pdf_document(
    pdf_path: str,
    min_digital_chars: int = 50,
    ocr_dpi: int = 200,
    ocr_lang: str = "eng",
    ocr_psm: str = "--psm 6",
) -> List[Dict[str, Any]]:
    """
    Stage 1 entry point: Parses a PDF file page by page.
    Returns a list of page dictionaries:
      - page_num (int, 1-indexed)
      - raw_text (str)
      - tables_markdown (List[str])
      - headings (List[str])
      - parse_method ('digital' or 'ocr')
      - char_count (int)
    """
    pages_output: List[Dict[str, Any]] = []

    fitz_doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)

    try:
        for idx in range(len(fitz_doc)):
            page_num = idx + 1
            fitz_page = fitz_doc[idx]
            plumber_page = plumber_doc.pages[idx]

            # 1. Extract digital text with two-column awareness
            digital_text = _extract_column_sorted_text(fitz_page)
            alnum_count = sum(1 for c in digital_text if c.isalnum())

            # 2. Extract any structured bordered tables via pdfplumber
            tables_md: List[str] = []
            raw_tables = plumber_page.extract_tables() or []
            for tbl in raw_tables:
                md_tbl = _format_table_as_markdown(tbl)
                if md_tbl:
                    tables_md.append(md_tbl)

            # 3. Check OCR fallback threshold
            if alnum_count < min_digital_chars:
                parse_method = "ocr"
                pix = fitz_page.get_pixmap(dpi=ocr_dpi, colorspace=fitz.csGRAY)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_text = pytesseract.image_to_string(
                    img,
                    lang=ocr_lang,
                    config=ocr_psm,
                )
            else:
                parse_method = "digital"
                page_text = digital_text

            # Append structured Markdown table(s) at the bottom of the page text if not already embedded
            if tables_md:
                page_text = page_text.rstrip() + "\n\n" + "\n\n".join(tables_md)

            headings = _extract_headings_from_text(page_text)

            pages_output.append(
                {
                    "page_num": page_num,
                    "raw_text": page_text,
                    "tables_markdown": tables_md,
                    "headings": headings,
                    "parse_method": parse_method,
                    "char_count": len(page_text),
                }
            )
    finally:
        fitz_doc.close()
        plumber_doc.close()

    return pages_output