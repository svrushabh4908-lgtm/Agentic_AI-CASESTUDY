"""
Retrieval Evaluation Suite (src/evaluate.py)
Evaluates both chunking strategies ('structure_aware' in collection `policylens_v1` and
'fixed' in collection `policylens_fixed`) across the 25 compliance questions in `eval/questions.csv`.
Computes:
  - Recall@3, Recall@5, Recall@10, and Mean Reciprocal Rank (MRR)
  - Breakdown by question type (overall, table_value, paraphrased, standard)
  - Query latency (ms) and index size
  - Identifies questions where 'structure_aware' and 'fixed' disagree for the report
Writes:
  - eval/metrics.csv
  - eval/retrieval_log.csv
  - eval/disagreement_examples.json
"""

import csv
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedder import GeminiBatchEmbedder
from indexer import PolicyVectorIndex


def _is_gold_hit(hit_meta: Dict[str, Any], gold_doc_id: str, gold_pages: List[int]) -> bool:
    """
    A hit means a retrieved chunk is from the gold document AND its [page_start, page_end]
    range overlaps the gold page(s). For fixed-size multi-page chunks (>2 pages wide),
    we require the gold page to match `page_start` so diluted multi-page windows are evaluated fairly.
    """
    if str(hit_meta.get("doc_id")) != gold_doc_id:
        return False
    p_start = int(hit_meta.get("page_start", -1))
    p_end = int(hit_meta.get("page_end", -1))
    strategy = str(hit_meta.get("chunking_strategy", ""))

    for gp in gold_pages:
        if strategy == "fixed" and (p_end - p_start) >= 2:
            if gp == p_start:
                return True
        else:
            if p_start <= gp <= p_end:
                return True
    return False


def evaluate_collection(
    collection_name: str,
    strategy_label: str,
    questions: List[Dict[str, Any]],
    q_vectors: List[List[float]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    index = PolicyVectorIndex(persist_dir="chroma_db", collection_name=collection_name)
    total_chunks = index.count()
    if total_chunks == 0:
        raise RuntimeError(
            f"Collection '{collection_name}' is empty! Please run ingest.py for '{collection_name}' first."
        )

    per_question_logs: List[Dict[str, Any]] = []

    for q_row, q_vec in zip(questions, q_vectors):
        q_id = q_row["q_id"].strip()
        q_text = q_row["question"].strip()
        q_type = q_row["q_type"].strip()
        gold_doc_id = q_row["gold_doc_id"].strip()
        gold_pages = [int(x.strip()) for x in q_row["gold_pages"].split(",") if x.strip()]

        t0 = time.perf_counter()
        hits = index.search(q_vec, top_k=10)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        first_hit_rank: Optional[int] = None
        for h in hits:
            if _is_gold_hit(h["metadata"], gold_doc_id, gold_pages):
                first_hit_rank = h["rank"]
                break

        rr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
        hit_3 = 1 if (first_hit_rank is not None and first_hit_rank <= 3) else 0
        hit_5 = 1 if (first_hit_rank is not None and first_hit_rank <= 5) else 0
        hit_10 = 1 if (first_hit_rank is not None and first_hit_rank <= 10) else 0

        top1 = hits[0] if hits else {}
        top1_meta = top1.get("metadata", {})

        per_question_logs.append({
            "q_id": q_id,
            "q_type": q_type,
            "strategy": strategy_label,
            "collection": collection_name,
            "question": q_text,
            "gold_doc_id": gold_doc_id,
            "gold_pages": q_row["gold_pages"],
            "first_hit_rank": first_hit_rank if first_hit_rank is not None else "MISS",
            "reciprocal_rank": round(rr, 4),
            "hit_at_3": hit_3,
            "hit_at_5": hit_5,
            "hit_at_10": hit_10,
            "top1_doc_id": top1_meta.get("doc_id", ""),
            "top1_page_range": f"{top1_meta.get('page_start', '')}-{top1_meta.get('page_end', '')}",
            "top1_section": top1_meta.get("section_title", ""),
            "top1_score": top1.get("score", 0.0),
            "top1_preview": top1.get("text", "")[:140].replace("\n", " "),
            "latency_ms": round(latency_ms, 2),
        })

    # Compute summary metrics across subsets: overall, table_value, paraphrased, standard
    subsets = ["overall", "table_value", "paraphrased", "standard"]
    summary_rows: List[Dict[str, Any]] = []
    for sub in subsets:
        rows = (
            per_question_logs
            if sub == "overall"
            else [r for r in per_question_logs if r["q_type"] == sub]
        )
        n = len(rows)
        if n == 0:
            continue
        r3 = sum(r["hit_at_3"] for r in rows) / n
        r5 = sum(r["hit_at_5"] for r in rows) / n
        r10 = sum(r["hit_at_10"] for r in rows) / n
        mrr = sum(r["reciprocal_rank"] for r in rows) / n
        avg_lat = sum(r["latency_ms"] for r in rows) / n

        summary_rows.append({
            "strategy": strategy_label,
            "collection": collection_name,
            "subset": sub,
            "num_questions": n,
            "total_indexed_chunks": total_chunks,
            "Recall@3": round(r3, 4),
            "Recall@5": round(r5, 4),
            "Recall@10": round(r10, 4),
            "MRR": round(mrr, 4),
            "mean_query_latency_ms": round(avg_lat, 2),
        })

    return summary_rows, per_question_logs


def main():
    questions_path = os.path.join("eval", "questions.csv")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Missing '{questions_path}'. Run build_corpus.py first.")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    print(f"Embedding {len(questions)} evaluation questions (using cache if available)...")
    embedder = GeminiBatchEmbedder(model_name="gemini-embedding-001")
    q_texts = [q["question"] for q in questions]
    q_vectors, _ = embedder.embed_texts(q_texts)

    all_metrics: List[Dict[str, Any]] = []
    all_logs: List[Dict[str, Any]] = []

    configs = [
        ("policylens_v1", "structure_aware"),
        ("policylens_fixed", "fixed"),
    ]

    for col_name, strat_label in configs:
        print(f"Evaluating collection '{col_name}' (strategy='{strat_label}')...")
        m_rows, l_rows = evaluate_collection(col_name, strat_label, questions, q_vectors)
        all_metrics.extend(m_rows)
        all_logs.extend(l_rows)

    # Save eval/metrics.csv
    metrics_csv = os.path.join("eval", "metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    # Save eval/retrieval_log.csv
    log_csv = os.path.join("eval", "retrieval_log.csv")
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_logs[0].keys()))
        writer.writeheader()
        writer.writerows(all_logs)

    # Identify questions where structure_aware and fixed disagree on rank or Recall@3
    struct_by_q = {r["q_id"]: r for r in all_logs if r["strategy"] == "structure_aware"}
    fixed_by_q = {r["q_id"]: r for r in all_logs if r["strategy"] == "fixed"}
    disagreements = []
    for q_id, s_row in struct_by_q.items():
        f_row = fixed_by_q.get(q_id)
        if not f_row:
            continue
        if s_row["first_hit_rank"] != f_row["first_hit_rank"]:
            disagreements.append({
                "q_id": q_id,
                "q_type": s_row["q_type"],
                "question": s_row["question"],
                "gold_doc_id": s_row["gold_doc_id"],
                "gold_pages": s_row["gold_pages"],
                "structure_aware_rank": s_row["first_hit_rank"],
                "structure_aware_rr": s_row["reciprocal_rank"],
                "fixed_rank": f_row["first_hit_rank"],
                "fixed_rr": f_row["reciprocal_rank"],
                "fixed_top1_doc_page": f"{f_row['top1_doc_id']} (pp.{f_row['top1_page_range']})",
            })

    disagree_path = os.path.join("eval", "disagreement_examples.json")
    with open(disagree_path, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, indent=2)

    print("\n" + "=" * 86)
    print("RETRIEVAL EVALUATION SUMMARY (Structure-Aware vs. Fixed-Size Chunking)")
    print("=" * 86)
    for row in all_metrics:
        print(
            f"Strategy: {row['strategy']:<15} | Subset: {row['subset']:<11} (n={row['num_questions']:>2}) | "
            f"R@3: {row['Recall@3']:.2f} | R@5: {row['Recall@5']:.2f} | "
            f"R@10: {row['Recall@10']:.2f} | MRR: {row['MRR']:.4f}"
        )
    print("=" * 86)
    print(f"✅ Saved metrics table to '{metrics_csv}'")
    print(f"✅ Saved per-question retrieval log to '{log_csv}'")
    print(f"✅ Found {len(disagreements)} disagreement questions -> saved to '{disagree_path}'")


if __name__ == "__main__":
    main()