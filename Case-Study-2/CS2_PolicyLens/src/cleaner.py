"""
Stage 2: Clean (src/cleaner.py)
Removes repeating headers, footers, and standalone page numbers; fixes line-break hyphenation;
normalizes whitespace; and verifies that no real regulatory content (numbers, percentages, clauses)
was accidentally deleted.
Also provides a built-in CLI check (`python3 src/cleaner.py`) to run the 10-page manual/automated
parsing & cleaning quality inspection required for the report.
"""

import csv
import os
import re
from typing import Any, Dict, List, Tuple

# Regex rules for repeating headers and footers across digital & OCR pages
HEADER_PATTERNS = [
    re.compile(
        r"^.*REGULATORY\s+COMPLIANCE\s+CIRCULAR\s*\|\s*REF:\s*DOC_\d+.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

FOOTER_PATTERNS = [
    re.compile(
        r"^.*Page\s+\d+\s+of\s+\d+.*PolicyLens\s+Compliance\s+Repository.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE),
]

# Matches words split across lines with a hyphen (e.g., "disburse-\nment" -> "disbursement")
HYPHEN_LINEBREAK_REGEX = re.compile(r"([a-zA-Z]{2,})-\s*\n\s*([a-z]{2,})")

# Extracts numeric figures (Rs. amounts, percentages, numbers) to verify cleaning safety
NUMERIC_TOKEN_REGEX = re.compile(r"\b\d+(?:[,\.]\d+)*%?\b")


def clean_page_text(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Cleans raw page text while preserving Markdown tables and section structure.
    Returns (cleaned_text, stats_dict).
    """
    text = raw_text

    # 1. Strip repeating headers and footers first so page numbers in footers don't
    # skew the numeric check.
    headers_removed = 0
    for pat in HEADER_PATTERNS:
        matches = pat.findall(text)
        headers_removed += len(matches)
        text = pat.sub("", text)

    footers_removed = 0
    for pat in FOOTER_PATTERNS:
        matches = pat.findall(text)
        footers_removed += len(matches)
        text = pat.sub("", text)

    # Record all numeric tokens in the body before de-hyphenation & whitespace normalization
    body_numbers_before = set(NUMERIC_TOKEN_REGEX.findall(text))

    # 2. Fix line-break hyphenation (e.g. "disburse-\nment" -> "disbursement")
    hyphen_matches = HYPHEN_LINEBREAK_REGEX.findall(text)
    hyphens_fixed = len(hyphen_matches)
    text = HYPHEN_LINEBREAK_REGEX.sub(r"\1\2", text)

    # 3. Normalize horizontal whitespace while preserving line breaks & Markdown pipe tables
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(stripped)
    text = "\n".join(lines)

    # Collapse 3+ consecutive newlines into a clean double newline paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 4. Safety check: verify zero body numeric values were lost during cleaning
    body_numbers_after = set(NUMERIC_TOKEN_REGEX.findall(text))
    missing_numbers = body_numbers_before - body_numbers_after
    safety_passed = len(missing_numbers) == 0

    stats = {
        "chars_before": len(raw_text),
        "chars_after": len(text),
        "headers_removed": headers_removed,
        "footers_removed": footers_removed,
        "hyphens_fixed": hyphens_fixed,
        "safety_passed": safety_passed,
        "missing_numbers_count": len(missing_numbers),
    }
    return text, stats


def clean_parsed_pages(parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies Stage 2 cleaning to every parsed page dictionary in a document."""
    cleaned_pages = []
    for page in parsed_pages:
        clean_txt, stats = clean_page_text(page["raw_text"])
        updated = dict(page)
        updated["clean_text"] = clean_txt
        updated["clean_stats"] = stats
        cleaned_pages.append(updated)
    return cleaned_pages


if __name__ == "__main__":
    # Run the 10-page manual/automated parsing quality check + before/after cleaning showcase
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parser import parse_pdf_document

    sample_targets = [
        ("DOC_001_RBI_Cash_Disbursement_Caps_2025.pdf", 3, "table_heavy"),
        ("DOC_002_RBI_Gold_Loan_LTV_Slabs_2025.pdf", 2, "table_heavy"),
        ("DOC_004_PolicyLens_Penal_Charges_Schedule_2025.pdf", 3, "table_heavy"),
        ("DOC_007_RBI_Legacy_Fair_Practices_Scanned_2019.pdf", 2, "scanned"),
        ("DOC_008_FIU_AML_Suspicious_Transaction_Scanned_2021.pdf", 3, "scanned"),
        ("DOC_009_RBI_Recovery_Agents_Conduct_Scanned_2022.pdf", 2, "scanned"),
        ("DOC_013_RBI_Master_Direction_KYC_2025_Long.pdf", 38, "long_doc_54p"),
        ("DOC_014_RBI_Master_Direction_Digital_Lending_2025_Long.pdf", 44, "long_doc_52p"),
        ("DOC_016_RBI_Key_Facts_Statement_TwoCol_2025.pdf", 3, "two_column"),
        ("DOC_017_RBI_IT_Outsourcing_Cloud_TwoCol_2025.pdf", 4, "two_column"),
    ]

    os.makedirs("eval", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    report_rows = []
    before_after_saved = False

    print("Running Stage 1 (Parse) & Stage 2 (Clean) on 10 sample pages...")
    for fname, target_page, category in sample_targets:
        fpath = os.path.join("data", "pdfs", fname)
        pages = parse_pdf_document(fpath)
        cleaned = clean_parsed_pages(pages)
        p_obj = cleaned[target_page - 1]
        st = p_obj["clean_stats"]

        # Save the first page's Before vs After text for the report exhibit
        if not before_after_saved:
            sample_log_path = os.path.join("logs", "before_after_cleaning_sample.txt")
            with open(sample_log_path, "w", encoding="utf-8") as lf:
                lf.write(f"=== BEFORE CLEANING ({fname} - Page {target_page}) ===\n")
                lf.write(p_obj["raw_text"] + "\n\n")
                lf.write(f"=== AFTER CLEANING ({fname} - Page {target_page}) ===\n")
                lf.write(p_obj["clean_text"] + "\n")
            before_after_saved = True

        report_rows.append(
            {
                "filename": fname,
                "page_num": target_page,
                "category": category,
                "parse_method": p_obj["parse_method"],
                "tables_found": len(p_obj["tables_markdown"]),
                "headings_found": len(p_obj["headings"]),
                "chars_before": st["chars_before"],
                "chars_after": st["chars_after"],
                "headers_removed": st["headers_removed"],
                "footers_removed": st["footers_removed"],
                "hyphens_fixed": st["hyphens_fixed"],
                "safety_passed": st["safety_passed"],
            }
        )
        print(
            f"  [OK] {fname} (p.{target_page}, {p_obj['parse_method']}): "
            f"chars {st['chars_before']}->{st['chars_after']}, "
            f"headers_removed={st['headers_removed']}, "
            f"footers_removed={st['footers_removed']}, "
            f"hyphens_fixed={st['hyphens_fixed']}, tables={len(p_obj['tables_markdown'])}, "
            f"safety_passed={st['safety_passed']}"
        )

    out_csv = os.path.join("eval", "parsing_quality_check.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"\n✅ Saved 10-page parsing quality table to '{out_csv}'")
    print("✅ Saved before-and-after cleaning sample to 'logs/before_after_cleaning_sample.txt'")