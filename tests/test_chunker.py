import re
from types import SimpleNamespace

import pytest

from app.ingestion import chunker
from app.ingestion.parser import Section


class WordTokenizer:
    _token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return [match.group() for match in self._token_pattern.finditer(text)]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[tuple[int, int]]]:
        del add_special_tokens, return_offsets_mapping
        return {
            "offset_mapping": [
                (match.start(), match.end())
                for match in self._token_pattern.finditer(text)
            ]
        }


@pytest.fixture
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> WordTokenizer:
    tokenizer = WordTokenizer()
    monkeypatch.setattr(chunker, "_get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        chunker,
        "get_settings",
        lambda: SimpleNamespace(
            parent_tokens=30,
            child_tokens=90,
            child_overlap=12,
        ),
    )
    return tokenizer


def test_tables_stay_one_parent_and_children_repeat_header(fake_tokenizer) -> None:
    table = "\n".join(
        [
            "| Code | Meaning |",
            "| --- | --- |",
            *[
                f"| ERR-{number:03} | Fault code number {number} |"
                for number in range(20)
            ],
        ]
    )
    section = Section(
        heading_path="Chiller Manual > Fault Codes",
        page_start=3,
        text_md=table,
        is_table=True,
    )

    parents = chunker.build_parents([section])
    children = chunker.build_children(parents[0], "tenant_alpha", "manual.pdf")
    breadcrumb = (
        "[Doc: manual.pdf | Section: Chiller Manual > Fault Codes | "
        "Tenant: tenant_alpha]"
    )

    assert len(parents) == 1
    assert parents[0].text_md == table
    assert len(children) > 1
    assert all(child.startswith(breadcrumb) for child in children)
    assert all("| Code | Meaning |\n| --- | --- |" in child for child in children)
    assert all(len(fake_tokenizer.encode(child)) <= 90 for child in children)
    for number in range(20):
        assert any(f"ERR-{number:03}" in child for child in children)


def test_text_children_have_breadcrumb_limit_and_token_overlap(fake_tokenizer) -> None:
    section = Section(
        heading_path="Chiller Manual > Maintenance",
        page_start=1,
        text_md="\n\n".join(
            f"Step {number} checks valve pressure and confirms normal operation."
            for number in range(30)
        ),
        is_table=False,
    )
    parent = chunker.Parent(
        heading_path=section.heading_path,
        page_start=section.page_start,
        text_md=section.text_md,
        is_table=False,
    )
    children = chunker.build_children(parent, "tenant_alpha", "manual.pdf")
    breadcrumb = (
        "[Doc: manual.pdf | Section: Chiller Manual > Maintenance | "
        "Tenant: tenant_alpha]"
    )
    child_bodies = [child.split("\n", maxsplit=1)[1] for child in children]

    assert len(children) > 1
    assert all(child.startswith(breadcrumb) for child in children)
    assert all(len(fake_tokenizer.encode(child)) <= 90 for child in children)
    assert all(len(fake_tokenizer.encode(child)) <= 800 for child in children)
    assert all(
        fake_tokenizer.encode(previous)[-12:] == fake_tokenizer.encode(following)[:12]
        for previous, following in zip(child_bodies, child_bodies[1:])
    )


def test_large_text_sections_are_parent_chunked(fake_tokenizer) -> None:
    section = Section(
        heading_path="Chiller Manual",
        page_start=2,
        text_md="\n\n".join(
            f"Paragraph {number} covers inspection and scheduled maintenance."
            for number in range(12)
        ),
        is_table=False,
    )

    parents = chunker.build_parents([section])

    assert len(parents) > 1
    assert all(len(fake_tokenizer.encode(parent.text_md)) <= 30 for parent in parents)
    assert all(parent.heading_path == section.heading_path for parent in parents)
