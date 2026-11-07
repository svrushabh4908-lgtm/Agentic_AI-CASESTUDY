"""
Stage 4: Enrich (src/enricher.py)
Computes SHA-256 file hashes for idempotent re-runs / version control and attaches all required
metadata fields to every chunk:
  doc_id, title, issuer, issue_date, page_start, page_end, section_title, source_hash,
  chunk_id, chunking_strategy, token_count
"""

import hashlib
import json
import os
from typing import Dict, Any, List


def compute_file_sha256(file_path: str) -> str:
    """Computes the SHA-256 hex digest of a PDF file for idempotency & version tracking."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(65536)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def enrich_chunks(
    raw_chunks: List[Dict[str, Any]],
    manifest_record: Dict[str, Any],
    pdf_path: str,
) -> List[Dict[str, Any]]:
    """
    Attaches complete document and provenance metadata to each chunk dictionary.
    Returns a list of enriched chunk dicts ready for embedding and vector indexing.
    """
    source_hash = compute_file_sha256(pdf_path)
    doc_id = str(manifest_record["doc_id"]).strip()
    title = str(manifest_record["title"]).strip()
    issuer = str(manifest_record["issuer"]).strip()
    issue_date = str(manifest_record["issue_date"]).strip()
    doc_type = str(manifest_record.get("doc_type", "text_native")).strip()
    url = str(manifest_record.get("url", "")).strip()

    enriched_list: List[Dict[str, Any]] = []
    for idx, ch in enumerate(raw_chunks, start=1):
        strategy = ch.get("chunking_strategy", "structure_aware")
        chunk_id = f"{doc_id}_{strategy}_c{idx:04d}"

        metadata = {
            "doc_id": doc_id,
            "title": title,
            "issuer": issuer,
            "issue_date": issue_date,
            "page_start": int(ch["page_start"]),
            "page_end": int(ch["page_end"]),
            "section_title": str(ch.get("section_title", "General Section")),
            "source_hash": source_hash,
            "doc_type": doc_type,
            "url": url,
            "chunking_strategy": strategy,
            "token_count": int(ch.get("token_count", 0)),
        }

        enriched_list.append({
            "chunk_id": chunk_id,
            "text": ch["text"],
            "metadata": metadata,
        })

    return enriched_list


if __name__ == "__main__":
    # Quick verification test for Stage 3 (Chunk) and Stage 4 (Enrich)
    import csv
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parser import parse_pdf_document
    from cleaner import clean_parsed_pages
    from chunker import chunk_document

    with open(os.path.join("data", "manifest.csv"), "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    sample_row = manifest_rows[0]  # DOC_001 (table_heavy)
    pdf_file = os.path.join("data", "pdfs", sample_row["filename"])

    parsed = parse_pdf_document(pdf_file)
    cleaned = clean_parsed_pages(parsed)

    fixed_chunks = enrich_chunks(
        chunk_document(cleaned, strategy="fixed", doc_title=sample_row["title"]),
        sample_row,
        pdf_file,
    )
    struct_chunks = enrich_chunks(
        chunk_document(cleaned, strategy="structure_aware", doc_title=sample_row["title"]),
        sample_row,
        pdf_file,
    )

    print(f"✅ Tested Stage 3 & Stage 4 on {sample_row['filename']}:")
    print(f"  - Fixed-size chunks produced:      {len(fixed_chunks)} chunks")
    print(f"  - Structure-aware chunks produced: {len(struct_chunks)} chunks")
    print("\nSample Enriched Metadata Record (Structure-Aware Key Page 3):")
    key_chunk = [c for c in struct_chunks if c["metadata"]["page_start"] == 3][0]
    print(json.dumps(key_chunk["metadata"], indent=2))