from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.ingestion.parser import Section

_TOKENIZER_NAME = "BAAI/bge-m3"
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_INSTRUCTION_LIKE_TEXT = re.compile(
    r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions?\b"
    r"|\breveal\b|\bsystem\s+prompt\b",
    re.IGNORECASE,
)


def scan_chunk(text: str) -> bool:
    """Return True when a chunk contains instruction-like text."""
    return _INSTRUCTION_LIKE_TEXT.search(text) is not None


def _is_table_separator(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)


@dataclass(frozen=True)
class Parent:
    heading_path: str
    page_start: int
    text_md: str
    is_table: bool


@lru_cache(maxsize=1)
def _get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_TOKENIZER_NAME, use_fast=True)


def _token_count(text: str) -> int:
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


def _token_offsets(text: str) -> list[tuple[int, int]]:
    encoded = _get_tokenizer()(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return [
        tuple(offset) for offset in encoded["offset_mapping"] if offset[1] > offset[0]
    ]


def build_parents(sections: list[Section]) -> list[Parent]:
    """Create parent chunks, retaining every table as one indivisible parent."""
    limit = get_settings().parent_tokens
    if limit <= 0:
        raise ValueError("PARENT_TOKENS must be greater than zero")

    parents: list[Parent] = []
    for section in sections:
        if section.is_table or _token_count(section.text_md) <= limit:
            parents.append(
                Parent(
                    heading_path=section.heading_path,
                    page_start=section.page_start,
                    text_md=section.text_md,
                    is_table=section.is_table,
                )
            )
            continue

        for chunk in _chunk_text(section.text_md, limit, overlap=0, prefix=""):
            parents.append(
                Parent(
                    heading_path=section.heading_path,
                    page_start=section.page_start,
                    text_md=chunk,
                    is_table=False,
                )
            )
    return parents


def build_children(
    parent: Parent | Section, tenant_id: str, doc_name: str
) -> list[str]:
    """Build breadcrumb-prefixed child chunks using the BGE-M3 tokenizer."""
    settings = get_settings()
    limit = min(settings.child_tokens, 800)
    overlap = settings.child_overlap
    breadcrumb = (
        f"[Doc: {doc_name} | Section: {parent.heading_path} | Tenant: {tenant_id}]"
    )
    prefix = breadcrumb + "\n"

    if limit <= 0:
        raise ValueError("CHILD_TOKENS must be greater than zero")
    if overlap < 0:
        raise ValueError("CHILD_OVERLAP cannot be negative")
    if _token_count(prefix) >= limit:
        raise ValueError("The breadcrumb uses all available child tokens")

    if parent.is_table:
        bodies = _table_children(parent.text_md, prefix, limit)
    else:
        bodies = _chunk_text(parent.text_md, limit, overlap, prefix)
    return [prefix + body.strip() for body in bodies]


def _fits(text: str, prefix: str, limit: int) -> bool:
    return _token_count(prefix + text) <= limit


def _split_at_boundaries(text: str, boundary: re.Pattern[str]) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in boundary.finditer(text):
        parts.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return [part for part in parts if part]


def _hard_split(text: str, prefix: str, limit: int) -> list[str]:
    offsets = _token_offsets(text)
    if not offsets:
        return [text] if _fits(text, prefix, limit) else []

    chunks: list[str] = []
    token_start = 0
    while token_start < len(offsets):
        low = token_start + 1
        high = len(offsets)
        best_end: int | None = None
        while low <= high:
            token_end = (low + high) // 2
            char_start = offsets[token_start][0]
            char_end = (
                offsets[token_end][0]
                if token_end < len(offsets)
                else offsets[token_end - 1][1]
            )
            candidate = text[char_start:char_end]
            if _fits(candidate, prefix, limit):
                best_end = token_end
                low = token_end + 1
            else:
                high = token_end - 1

        if best_end is None:
            raise ValueError("The child token limit is too small for one BGE-M3 token")

        char_start = offsets[token_start][0]
        char_end = (
            offsets[best_end][0]
            if best_end < len(offsets)
            else offsets[best_end - 1][1]
        )
        chunk = text[char_start:char_end].strip()
        if chunk:
            chunks.append(chunk)
        token_start = best_end
    return chunks


def _bounded_units(text: str, prefix: str, limit: int) -> list[str]:
    units: list[str] = []
    for paragraph in _split_at_boundaries(text, _PARAGRAPH_BREAK):
        if _fits(paragraph, prefix, limit):
            units.append(paragraph)
            continue

        sentences = _split_at_boundaries(paragraph, _SENTENCE_BREAK)
        if len(sentences) <= 1:
            sentences = [paragraph]
        for sentence in sentences:
            if _fits(sentence, prefix, limit):
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, prefix, limit))
    return units


def _overlap_plus_unit(
    previous: str,
    unit: str,
    overlap: int,
    prefix: str,
    limit: int,
) -> str:
    offsets = _token_offsets(previous)
    if not offsets or overlap <= 0:
        return unit.lstrip()

    first_candidate = max(0, len(offsets) - overlap)
    for start_token in range(first_candidate, len(offsets)):
        overlap_text = previous[offsets[start_token][0] :].lstrip()
        candidate = overlap_text + unit
        if _fits(candidate, prefix, limit):
            return candidate
    return unit.lstrip()


def _chunk_text(text: str, limit: int, overlap: int, prefix: str) -> list[str]:
    if not text.strip():
        return []
    if not _fits("", prefix, limit):
        raise ValueError("The breadcrumb uses all available child tokens")

    prefix_tokens = _token_count(prefix)
    effective_overlap = min(overlap, max(0, limit - prefix_tokens - 1))
    unit_limit = limit - effective_overlap
    units = _bounded_units(text, prefix, unit_limit)
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = current + unit
        if _fits(candidate, prefix, limit):
            current = candidate
            continue

        if current.strip():
            chunks.append(current.strip())
        current = _overlap_plus_unit(
            current,
            unit,
            effective_overlap,
            prefix,
            limit,
        )
        if not _fits(current, prefix, limit):
            raise ValueError("A text unit exceeds the configured token limit")

    if current.strip():
        chunks.append(current.strip())
    return chunks


def _table_children(text: str, prefix: str, limit: int) -> list[str]:
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index in range(len(lines) - 1)
            if "|" in lines[index] and _is_table_separator(lines[index + 1])
        ),
        None,
    )
    if header_index is None:
        if _fits(text, prefix, limit):
            return [text.strip()]
        raise ValueError("Oversized table is not a Markdown pipe table")

    lead = "\n".join(lines[:header_index]).strip()
    header = "\n".join(lines[header_index : header_index + 2])
    rows = [line for line in lines[header_index + 2 :] if line.strip()]
    table_head = f"{lead}\n{header}".strip() if lead else header
    if not _fits(table_head, prefix, limit):
        raise ValueError("The table header exceeds the configured child token limit")

    children: list[str] = []
    group: list[str] = []
    for row in rows:
        candidate = "\n".join([table_head, *group, row])
        if _fits(candidate, prefix, limit):
            group.append(row)
            continue

        if not group:
            raise ValueError("A table row exceeds the configured child token limit")
        children.append("\n".join([table_head, *group]))
        group = [row]
        if not _fits("\n".join([table_head, row]), prefix, limit):
            raise ValueError("A table row exceeds the configured child token limit")

    if group:
        children.append("\n".join([table_head, *group]))
    elif not children:
        children.append(table_head)
    return children
