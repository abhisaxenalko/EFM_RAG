from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PAGE_BREAK = "<!-- __efm_page_break__ -->"
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_HTML_TABLE_START_RE = re.compile(r"^\s*<table\b", re.IGNORECASE)
_HTML_TABLE_END_RE = re.compile(r"</table\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class Section:
    heading_path: str
    page_start: int
    text_md: str
    is_table: bool


def parse_document(path: str | Path) -> list[Section]:
    """Parse a local document into Markdown sections, retaining tables as tables.

    Markdown files use the lightweight parser directly. Other supported formats
    are converted locally by Docling; its import is delayed until it is needed so
    importing this module does not initialize models or perform network access.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    if source.suffix.lower() in {".md", ".markdown"}:
        markdown = source.read_text(encoding="utf-8-sig")
        return _split_markdown(markdown)

    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(source)
    try:
        markdown = result.document.export_to_markdown(
            page_break_placeholder=_PAGE_BREAK
        )
    finally:
        result.input.release()
    return _split_markdown(markdown)


def _split_markdown(markdown: str) -> list[Section]:
    sections: list[Section] = []
    heading_stack: list[str] = []
    page = 1
    text_lines: list[str] = []
    text_page = page
    fence_char: str | None = None

    def heading_path() -> str:
        return " > ".join(heading_stack)

    def flush_text() -> None:
        nonlocal text_lines
        text = "\n".join(text_lines).strip()
        if text:
            sections.append(
                Section(
                    heading_path=heading_path(),
                    page_start=text_page,
                    text_md=text,
                    is_table=False,
                )
            )
        text_lines = []

    def add_text(line: str) -> None:
        nonlocal text_page
        if line.strip() and not any(existing.strip() for existing in text_lines):
            text_page = page
        text_lines.append(line)

    def pipe_cells(line: str) -> list[str]:
        cells = line.strip().strip("|").split("|")
        return [cell.strip() for cell in cells]

    def is_table_delimiter(line: str) -> bool:
        cells = pipe_cells(line)
        return bool(cells) and all(
            re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells
        )

    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]

        if line.strip() == _PAGE_BREAK:
            flush_text()
            page += 1
            index += 1
            continue

        fence_match = _FENCE_RE.match(line)
        if fence_char is not None:
            add_text(line)
            if fence_match and fence_match.group(1)[0] == fence_char:
                fence_char = None
            index += 1
            continue
        if fence_match:
            fence_char = fence_match.group(1)[0]
            add_text(line)
            index += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_text()
            level = len(heading_match.group(1))
            del heading_stack[level - 1 :]
            heading_stack.append(heading_match.group(2).strip())
            index += 1
            continue

        if (
            index + 1 < len(lines)
            and "|" in line
            and is_table_delimiter(lines[index + 1])
        ):
            flush_text()
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index]:
                table_lines.append(lines[index])
                index += 1
            sections.append(
                Section(
                    heading_path=heading_path(),
                    page_start=page,
                    text_md="\n".join(table_lines).strip(),
                    is_table=True,
                )
            )
            continue

        if _HTML_TABLE_START_RE.match(line):
            flush_text()
            table_lines = [line]
            index += 1
            while index < len(lines):
                table_lines.append(lines[index])
                if _HTML_TABLE_END_RE.search(lines[index]):
                    index += 1
                    break
                index += 1
            sections.append(
                Section(
                    heading_path=heading_path(),
                    page_start=page,
                    text_md="\n".join(table_lines).strip(),
                    is_table=True,
                )
            )
            continue

        add_text(line)
        index += 1

    flush_text()
    return sections
