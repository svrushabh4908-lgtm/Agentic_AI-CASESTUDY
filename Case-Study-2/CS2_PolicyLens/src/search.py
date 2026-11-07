"""
Stage 7: Operate - Search CLI (src/search.py)
Required interface:
  python src/search.py "cash disbursement limit for secured loans" --top_k 5 --filter issuer=RBI
  # search output: rank, score, doc title, page range, section, first 200 characters of chunk
"""

import argparse
import os
import sys
import time
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedder import GeminiBatchEmbedder
from indexer import PolicyVectorIndex


def parse_filter_arg(filter_str: str) -> Optional[Dict[str, Any]]:
    """
    Parses filter strings such as 'issuer=RBI' or 'doc_id=DOC_001,issuer=RBI'
    into a ChromaDB metadata `where` dictionary.
    """
    if not filter_str or not filter_str.strip():
        return None
    parts = [p.strip() for p in filter_str.split(",") if "=" in p]
    if not parts:
        return None
    if len(parts) == 1:
        k, v = parts[0].split("=", 1)
        return {k.strip(): v.strip()}
    and_clauses = []
    for p in parts:
        k, v = p.split("=", 1)
        and_clauses.append({k.strip(): v.strip()})
    return {"$and": and_clauses}


def main():
    parser = argparse.ArgumentParser(description="PolicyLens Semantic Retrieval CLI")
    parser.add_argument("query", type=str, help="Compliance search question")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Optional metadata filter, e.g., issuer=RBI or doc_type=scanned",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="policylens_v1",
        help="ChromaDB collection to query",
    )
    args = parser.parse_args()

    where_filter = parse_filter_arg(args.filter)
    embedder = GeminiBatchEmbedder(model_name="gemini-embedding-001")
    index = PolicyVectorIndex(persist_dir="chroma_db", collection_name=args.collection)

    t0 = time.perf_counter()
    q_vec = embedder.embed_query(args.query)
    hits = index.search(q_vec, top_k=args.top_k, where_filter=where_filter)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print("=" * 90)
    print(f"Query      : {args.query}")
    print(f"Collection : {args.collection} | Top-K: {args.top_k} | Filter: {where_filter or 'None'} | Latency: {latency_ms:.1f} ms")
    print("=" * 90)

    if not hits:
        print("No matching passages found.")
        return

    for h in hits:
        meta = h["metadata"]
        p_start = meta.get("page_start")
        p_end = meta.get("page_end")
        page_range = f"p.{p_start}" if p_start == p_end else f"pp.{p_start}-{p_end}"
        snippet_200 = h["text"].replace("\n", " ").strip()[:200]

        print(
            f"Rank #{h['rank']} | Score: {h['score']:.4f} | "
            f"Doc: [{meta.get('doc_id')}] {meta.get('title')} (Issuer: {meta.get('issuer')})"
        )
        print(f"  Page Range : {page_range} | Section: {meta.get('section_title')}")
        print(f"  Preview    : {snippet_200}...")
        print("-" * 90)


if __name__ == "__main__":
    main()