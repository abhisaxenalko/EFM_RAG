import sys
from types import ModuleType

from app.ingestion.parser import parse_document


def test_markdown_table_is_one_section_and_keeps_header(tmp_path) -> None:
    source = tmp_path / "manual.md"
    source.write_text(
        """# Chiller Manual

## Fault Codes

| Code | Meaning |
| --- | --- |
| ERR-409B | Low flow |
""",
        encoding="utf-8",
    )

    sections = parse_document(source)
    tables = [section for section in sections if section.is_table]

    assert len(tables) == 1
    assert tables[0].heading_path == "Chiller Manual > Fault Codes"
    assert "| Code | Meaning |" in tables[0].text_md
    assert "| ERR-409B | Low flow |" in tables[0].text_md
    assert tables[0].page_start == 1


def test_heading_path_tracks_nested_sections(tmp_path) -> None:
    source = tmp_path / "nested.md"
    source.write_text(
        """# Chiller Manual

## Fault Codes

ERR-409B indicates low flow.

### Reset procedure

Restart the pump.
""",
        encoding="utf-8",
    )

    sections = parse_document(source)

    assert [section.heading_path for section in sections] == [
        "Chiller Manual > Fault Codes",
        "Chiller Manual > Fault Codes > Reset procedure",
    ]
    assert all(not section.is_table for section in sections)


def test_other_formats_use_docling_and_export_page_markers(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "manual.pdf"
    source.touch()
    markdown = (
        "# Manual\n\nFirst page.\n"
        + "<!-- __efm_page_break__ -->"
        + ("\n\n| Code | Meaning |\n| --- | --- |\n| E-1 | Fault |")
    )
    release_called = False
    exported_with_page_markers = False

    class FakeDocument:
        def export_to_markdown(self, *, page_break_placeholder: str) -> str:
            nonlocal exported_with_page_markers
            exported_with_page_markers = page_break_placeholder == (
                "<!-- __efm_page_break__ -->"
            )
            return markdown

    class FakeInput:
        def release(self) -> None:
            nonlocal release_called
            release_called = True

    class FakeResult:
        document = FakeDocument()
        input = FakeInput()

    class FakeConverter:
        def convert(self, path):
            assert path == source
            return FakeResult()

    docling_package = ModuleType("docling")
    docling_package.__path__ = []
    converter_module = ModuleType("docling.document_converter")
    converter_module.DocumentConverter = FakeConverter
    monkeypatch.setitem(sys.modules, "docling", docling_package)
    monkeypatch.setitem(sys.modules, "docling.document_converter", converter_module)

    sections = parse_document(source)

    assert [section.page_start for section in sections] == [1, 2]
    assert sections[1].is_table
    assert "| Code | Meaning |" in sections[1].text_md
    assert release_called
    assert exported_with_page_markers
