"""
Stage 7: Operate - Ingestion CLI (src/ingest.py)
Required interface:
  python src/ingest.py --input data/pdfs --manifest data/manifest.csv --collection policylens_v1
Supports:
  - End-to-end Parse -> Clean -> Chunk -> Enrich -> Embed -> Index
  - Idempotent re-runs: checks SHA-256 source_hash in ChromaDB; skips unchanged files (zero duplicates)
  - Delete-and-replace: if a file's SHA-256 hash changes, deletes old chunks and indexes the revised version
  - Detailed logging to both terminal and log file (e.g. logs/run1.log, logs/run2_rerun.log)
  - Automatic export of index evidence to evidence/index_evidence_<collection>.json
"""

import argparse
import csv
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parser import parse_pdf_document
from cleaner import clean_parsed_pages
from chunker import chunk_document
from enricher import compute_file_sha256, enrich_chunks
from embedder import GeminiBatchEmbedder
from indexer import PolicyVectorIndex


def run_ingestion(
    input_dir: str,
    manifest_path: str,
    collection_name: str,
    strategy: str = "structure_aware",
    log_file: str = "",
) -> None:
    os.makedirs("logs", exist_ok=True)
    if not log_file:
        log_file = os.path.join("logs", f"ingest_{collection_name}.log")

    log_lines = []

    def log(msg: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_lines.append(line)

    log("=" * 78)
    log(f"POLICYLENS INGESTION PIPELINE START | collection={collection_name} | strategy={strategy}")
    log(f"Input PDF dir: {input_dir} | Manifest: {manifest_path}")
    log("=" * 78)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at '{manifest_path}'")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))

    index = PolicyVectorIndex(persist_dir="chroma_db", collection_name=collection_name)
    embedder = GeminiBatchEmbedder(model_name="gemini-embedding-001", batch_size=64)

    count_before = index.count()
    files_processed = 0
    files_skipped = 0
    files_updated = 0
    files_failed = 0
    total_pages_parsed = 0
    total_ocr_pages = 0
    total_chunks_added = 0
    total_embed_sec = 0.0

    t_start = time.perf_counter()

    for row in manifest_rows:
        doc_id = row["doc_id"].strip()
        filename = row["filename"].strip()
        pdf_path = os.path.join(input_dir, filename)

        if not os.path.exists(pdf_path):
            log(f"[ERROR] {doc_id} ({filename}): File missing at '{pdf_path}'")
            files_failed += 1
            continue

        try:
            current_hash = compute_file_sha256(pdf_path)
            existing_hash = index.get_existing_doc_hash(doc_id)

            # Stage 7 Idempotency check: skip unchanged PDFs
            if existing_hash == current_hash:
                log(
                    f"[SKIP - UNCHANGED] {doc_id} ({filename}) | sha256={current_hash[:12]}... "
                    f"(already indexed, 0 duplicates added)"
                )
                files_skipped += 1
                continue

            # If hash differs from an existing version, delete old chunks first
            if existing_hash is not None and existing_hash != current_hash:
                removed = index.delete_document_chunks(doc_id)
                log(
                    f"[UPDATE - REPLACED] {doc_id} ({filename}) | hash changed "
                    f"{existing_hash[:8]}..->{current_hash[:8]}.. | removed {removed} old chunks"
                )
                files_updated += 1

            # Stage 1: Parse
            parsed_pages = parse_pdf_document(pdf_path)
            num_pages = len(parsed_pages)
            ocr_pages = sum(1 for p in parsed_pages if p["parse_method"] == "ocr")
            tables_found = sum(len(p["tables_markdown"]) for p in parsed_pages)

            # Stage 2: Clean
            cleaned_pages = clean_parsed_pages(parsed_pages)

            # Stage 3: Chunk
            raw_chunks = chunk_document(
                cleaned_pages,
                strategy=strategy,
                doc_title=row["title"],
            )

            # Stage 4: Enrich
            enriched_chunks = enrich_chunks(raw_chunks, row, pdf_path)

            # Stage 5: Embed
            texts_to_embed = [c["text"] for c in enriched_chunks]
            vectors, embed_sec = embedder.embed_texts(texts_to_embed)
            total_embed_sec += embed_sec

            # Stage 6: Index
            upserted = index.upsert_chunks(enriched_chunks, vectors)

            files_processed += 1
            total_pages_parsed += num_pages
            total_ocr_pages += ocr_pages
            total_chunks_added += upserted

            log(
                f"[INGESTED] {doc_id} | {filename} | pages={num_pages} (ocr={ocr_pages}, tables={tables_found}) "
                f"| chunks={upserted} | embed_time={embed_sec:.2f}s | hash={current_hash[:10]}..."
            )
        except Exception as exc:
            files_failed += 1
            log(f"[ERROR] Failed processing {doc_id} ({filename}): {exc}")

    elapsed_total = time.perf_counter() - t_start
    count_after = index.count()
    sec_per_100_pages = (elapsed_total / total_pages_parsed * 100.0) if total_pages_parsed > 0 else 0.0

    evidence_path = index.export_evidence(output_dir="evidence", sample_limit=5)

    log("-" * 78)
    log("INGESTION RUN SUMMARY")
    log(f"  Collection Name        : {collection_name}")
    log(f"  Chunking Strategy      : {strategy}")
    log(f"  Total Manifest Files   : {len(manifest_rows)}")
    log(f"  Files Newly Ingested   : {files_processed}")
    log(f"  Files Updated/Replaced : {files_updated}")
    log(f"  Files Skipped (No-Op)  : {files_skipped}")
    log(f"  Files Failed           : {files_failed}")
    log(f"  Total Pages Parsed     : {total_pages_parsed} (OCR pages: {total_ocr_pages})")
    log(f"  Chunks Added This Run  : {total_chunks_added}")
    log(f"  Collection Size Before : {count_before}")
    log(f"  Collection Size After  : {count_after}")
    log(f"  Total Elapsed Time     : {elapsed_total:.2f}s (Embed API time: {total_embed_sec:.2f}s)")
    if total_pages_parsed > 0:
        log(f"  Efficiency (per 100p)  : {sec_per_100_pages:.2f}s per 100 pages")
    log(f"  Evidence Exported To   : {evidence_path}")
    log("=" * 78)

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write("\n".join(log_lines) + "\n")
    print(f"\n✅ Saved full ingestion log to '{log_file}'")


def main():
    parser = argparse.ArgumentParser(description="PolicyLens PDF Ingestion Pipeline")
    parser.add_argument("--input", default="data/pdfs", help="Path to directory of PDFs")
    parser.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--collection", default="policylens_v1", help="ChromaDB collection name")
    parser.add_argument(
        "--strategy",
        default="structure_aware",
        choices=["structure_aware", "fixed"],
        help="Chunking strategy to use ('structure_aware' or 'fixed')",
    )
    parser.add_argument("--log_file", default="", help="Optional path for output log file")
    args = parser.parse_args()

    run_ingestion(
        input_dir=args.input,
        manifest_path=args.manifest,
        collection_name=args.collection,
        strategy=args.strategy,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()