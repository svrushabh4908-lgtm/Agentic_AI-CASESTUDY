"""
Stage 3: Chunk (src/chunker.py)
Implements two chunking strategies for comparative evaluation:
1. 'fixed': Fixed token window slicing with token overlap across page boundaries.
2. 'structure_aware': Splits along regulatory section/clause boundaries, keeps Markdown tables
   ([TABLE_START]...[TABLE_END]) intact with their governing clause, and preserves section titles.
"""

import re
from typing import Dict, Any, List
import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")

SECTION_SPLIT_REGEX = re.compile(
    r"(?=(?:^|\n)(?:Section|Clause|Chapter)\s+\d+(?:\.\d+)?[:\s]|(?:^|\n)\d+\.\d+\s+[A-Z])",
    re.MULTILINE,
)

HEADING_LINE_REGEX = re.compile(
    r"^(?:Section|Clause|Chapter)\s+\d+(?:\.\d+)?[:\s].*|^\d+\.\d+\s+[A-Z].*"
)


def count_tokens(text: str) -> int:
    """Returns token count using cl100k_base tokenizer."""
    return len(ENCODING.encode(text))


def _infer_section_title(text: str, fallback: str = "General Regulatory Provisions") -> str:
    """Extracts the primary section or clause heading from a passage."""
    for line in text.splitlines():
        s = line.strip()
        if HEADING_LINE_REGEX.match(s) and len(s) < 140:
            return s
    return fallback


def chunk_fixed_size(
    cleaned_pages: List[Dict[str, Any]],
    chunk_size_tokens: int = 160,
    chunk_overlap_tokens: int = 30,
) -> List[Dict[str, Any]]:
    """
    Strategy A: Fixed-size token chunking with overlap.
    Concatenates document pages into a single token stream while tracking exact source page
    per token, then slices into fixed windows. Because it ignores section/table boundaries,
    tables or multi-part clauses can be split across adjacent chunks.
    """
    token_stream: List[int] = []
    token_pages: List[int] = []
    page_headings: Dict[int, str] = {}

    for p in cleaned_pages:
        p_num = p["page_num"]
        txt = p.get("clean_text", p.get("raw_text", "")).strip()
        if not txt:
            continue
        page_headings[p_num] = (
            p["headings"][0] if p.get("headings") else f"Page {p_num} Provisions"
        )
        toks = ENCODING.encode(txt + "\n\n")
        token_stream.extend(toks)
        token_pages.extend([p_num] * len(toks))

    chunks: List[Dict[str, Any]] = []
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)
    total_tokens = len(token_stream)

    idx = 0
    while idx < total_tokens:
        end_idx = min(idx + chunk_size_tokens, total_tokens)
        sub_toks = token_stream[idx:end_idx]
        sub_pages = token_pages[idx:end_idx]

        chunk_text = ENCODING.decode(sub_toks).strip()
        if chunk_text:
            p_start = min(sub_pages)
            p_end = max(sub_pages)
            sec_title = _infer_section_title(
                chunk_text, fallback=page_headings.get(p_start, "General Section")
            )
            chunks.append({
                "text": chunk_text,
                "page_start": p_start,
                "page_end": p_end,
                "section_title": sec_title,
                "token_count": len(sub_toks),
                "chunking_strategy": "fixed",
            })
        if end_idx >= total_tokens:
            break
        idx += step

    return chunks


def chunk_structure_aware(
    cleaned_pages: List[Dict[str, Any]],
    min_chunk_tokens: int = 120,
    max_chunk_tokens: int = 650,
    doc_title: str = "",
) -> List[Dict[str, Any]]:
    """
    Strategy B: Structure-aware chunking by heading, clause, and intact table blocks.
    - Keeps every [TABLE_START]...[TABLE_END] table attached to its governing section clauses
    - Splits only on structural section/clause boundaries if a page exceeds max_chunk_tokens
    - Pads or merges short consecutive clauses on the same page to stay in the target token range
    """
    chunks: List[Dict[str, Any]] = []

    for p in cleaned_pages:
        p_num = p["page_num"]
        txt = p.get("clean_text", p.get("raw_text", "")).strip()
        if not txt:
            continue

        primary_heading = (
            p["headings"][0]
            if p.get("headings")
            else _infer_section_title(txt, fallback=f"Section {p_num}")
        )

        # Temporarily protect [TABLE_START]...[TABLE_END] so section splitting never cuts inside a table
        protected_tables: Dict[str, str] = {}

        def _protect_table(match: re.Match) -> str:
            key = f"__PROTECTED_TABLE_{len(protected_tables)}__"
            protected_tables[key] = match.group(0)
            return key

        safe_txt = re.sub(
            r"\[TABLE_START\].*?\[TABLE_END\]",
            _protect_table,
            txt,
            flags=re.DOTALL,
        )

        # If the whole page (with its clauses + table) fits within max_chunk_tokens, keep it cohesive
        total_p_tokens = count_tokens(txt)
        if total_p_tokens <= max_chunk_tokens:
            header_prefix = f"[{doc_title} | {primary_heading}]\n" if doc_title else f"[{primary_heading}]\n"
            full_passage = header_prefix + txt
            chunks.append({
                "text": full_passage,
                "page_start": p_num,
                "page_end": p_num,
                "section_title": primary_heading,
                "token_count": count_tokens(full_passage),
                "chunking_strategy": "structure_aware",
            })
            continue

        # Otherwise split along structural section/clause headings and re-attach protected tables
        raw_parts = [part.strip() for part in SECTION_SPLIT_REGEX.split(safe_txt) if part.strip()]
        restored_parts: List[str] = []
        for part in raw_parts:
            for k, v in protected_tables.items():
                part = part.replace(k, v)
            restored_parts.append(part)

        # Merge small clauses until they reach min_chunk_tokens without exceeding max_chunk_tokens
        buffer = ""
        buffer_heading = primary_heading
        for part in restored_parts:
            part_heading = _infer_section_title(part, fallback=buffer_heading)
            candidate = (buffer + "\n\n" + part).strip() if buffer else part
            if buffer and count_tokens(candidate) > max_chunk_tokens:
                header_prefix = f"[{doc_title} | {buffer_heading}]\n" if doc_title else f"[{buffer_heading}]\n"
                passage = header_prefix + buffer
                chunks.append({
                    "text": passage,
                    "page_start": p_num,
                    "page_end": p_num,
                    "section_title": buffer_heading,
                    "token_count": count_tokens(passage),
                    "chunking_strategy": "structure_aware",
                })
                buffer = part
                buffer_heading = part_heading
            else:
                if not buffer:
                    buffer_heading = part_heading
                buffer = candidate

        if buffer:
            header_prefix = f"[{doc_title} | {buffer_heading}]\n" if doc_title else f"[{buffer_heading}]\n"
            passage = header_prefix + buffer
            chunks.append({
                "text": passage,
                "page_start": p_num,
                "page_end": p_num,
                "section_title": buffer_heading,
                "token_count": count_tokens(passage),
                "chunking_strategy": "structure_aware",
            })

    return chunks


def chunk_document(
    cleaned_pages: List[Dict[str, Any]],
    strategy: str = "structure_aware",
    doc_title: str = "",
    fixed_size: int = 160,
    fixed_overlap: int = 30,
    min_struct_tokens: int = 120,
    max_struct_tokens: int = 650,
) -> List[Dict[str, Any]]:
    """Dispatches to either 'fixed' or 'structure_aware' chunking strategy."""
    if strategy == "fixed":
        return chunk_fixed_size(
            cleaned_pages,
            chunk_size_tokens=fixed_size,
            chunk_overlap_tokens=fixed_overlap,
        )
    elif strategy == "structure_aware":
        return chunk_structure_aware(
            cleaned_pages,
            min_chunk_tokens=min_struct_tokens,
            max_chunk_tokens=max_struct_tokens,
            doc_title=doc_title,
        )
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose 'fixed' or 'structure_aware'.")