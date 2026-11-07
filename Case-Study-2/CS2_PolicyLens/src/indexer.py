"""
Stage 6: Index (src/indexer.py)
Manages persistent ChromaDB collections with cosine similarity, idempotent hash checks,
delete-and-replace updates for revised circulars, metadata-filtered search, and index evidence export.
"""

import json
import os
from typing import Dict, Any, List, Optional
import chromadb


class PolicyVectorIndex:
    """Wrapper around persistent ChromaDB collection with idempotent document management."""

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        collection_name: str = "policylens_v1",
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_existing_doc_hash(self, doc_id: str) -> Optional[str]:
        """
        Checks if doc_id already exists in the collection.
        Returns the stored source_hash if present, or None if not yet indexed.
        """
        res = self.collection.get(
            where={"doc_id": doc_id},
            limit=1,
            include=["metadatas"],
        )
        metadatas = res.get("metadatas") or []
        if metadatas and len(metadatas) > 0:
            return metadatas[0].get("source_hash")
        return None

    def delete_document_chunks(self, doc_id: str) -> int:
        """Deletes all existing chunks for a doc_id (used when a circular is revised/updated)."""
        existing = self.collection.get(where={"doc_id": doc_id}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def upsert_chunks(
        self,
        enriched_chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> int:
        """Upserts a list of enriched chunks and their embedding vectors into ChromaDB."""
        if not enriched_chunks:
            return 0
        ids = [c["chunk_id"] for c in enriched_chunks]
        documents = [c["text"] for c in enriched_chunks]
        metadatas = [c["metadata"] for c in enriched_chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def count(self) -> int:
        """Returns total number of indexed chunks in the collection."""
        return self.collection.count()

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Queries the vector index using cosine similarity and optional metadata filter.
        Returns a ranked list of hit dictionaries.
        """
        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        res = self.collection.query(**query_kwargs)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]

        hits: List[Dict[str, Any]] = []
        for rank_idx, (cid, doc_text, meta, dist) in enumerate(
            zip(ids, docs, metas, dists), start=1
        ):
            # Chroma cosine distance = 1 - cosine_similarity
            cosine_score = round(1.0 - float(dist), 4)
            hits.append({
                "rank": rank_idx,
                "chunk_id": cid,
                "score": cosine_score,
                "text": doc_text,
                "metadata": meta,
            })
        return hits

    def export_evidence(self, output_dir: str = "evidence", sample_limit: int = 5) -> str:
        """
        Exports collection size and sample records with metadata to JSON for Deliverable #7.
        """
        os.makedirs(output_dir, exist_ok=True)
        total_chunks = self.count()
        sample = self.collection.get(
            limit=sample_limit,
            include=["documents", "metadatas"],
        )
        records = []
        for cid, doc, meta in zip(
            sample.get("ids", []),
            sample.get("documents", []),
            sample.get("metadatas", []),
        ):
            records.append({
                "chunk_id": cid,
                "metadata": meta,
                "document_preview_200chars": doc[:200].replace("\n", " "),
            })

        evidence_payload = {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_dir,
            "similarity_metric": "cosine",
            "total_indexed_chunks": total_chunks,
            "sample_records": records,
        }
        out_path = os.path.join(output_dir, f"index_evidence_{self.collection_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evidence_payload, f, indent=2)
        return out_path