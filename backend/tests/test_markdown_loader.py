from pathlib import Path

import pytest

from app.rag.markdown_loader import MarkdownLoader


def test_loads_markdown_file(tmp_path: Path) -> None:
    markdown_file = tmp_path / "document.md"
    markdown_file.write_text("# Document\nContent", encoding="utf-8")

    documents = MarkdownLoader(tmp_path).load()

    assert len(documents) == 1
    assert documents[0].content == "# Document\nContent"
    assert documents[0].filename == "document.md"
    assert documents[0].absolute_path == str(markdown_file.resolve())


def test_loads_markdown_files_recursively(tmp_path: Path) -> None:
    nested_directory = tmp_path / "topic" / "section"
    nested_directory.mkdir(parents=True)
    (nested_directory / "nested.md").write_text("Nested", encoding="utf-8")

    documents = MarkdownLoader(tmp_path).load()

    assert len(documents) == 1
    assert documents[0].filename == "nested.md"


def test_ignores_non_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "document.md").write_text("Markdown", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Text", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"image")

    documents = MarkdownLoader(tmp_path).load()

    assert [document.filename for document in documents] == ["document.md"]


def test_returns_path_relative_to_material_directory(tmp_path: Path) -> None:
    nested_directory = tmp_path / "java"
    nested_directory.mkdir()
    (nested_directory / "spring.md").write_text("Spring", encoding="utf-8")

    documents = MarkdownLoader(tmp_path).load()

    assert documents[0].relative_path == "java/spring.md"


def test_raises_error_when_material_directory_does_not_exist(
    tmp_path: Path,
) -> None:
    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Material directory does not exist"):
        MarkdownLoader(missing_directory).load()


def test_returns_empty_list_when_no_markdown_files_exist(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("Text", encoding="utf-8")

    assert MarkdownLoader(tmp_path).load() == []


def test_returns_documents_sorted_by_relative_path(tmp_path: Path) -> None:
    (tmp_path / "z.md").write_text("Z", encoding="utf-8")
    (tmp_path / "a.md").write_text("A", encoding="utf-8")

    documents = MarkdownLoader(tmp_path).load()

    assert [document.relative_path for document in documents] == ["a.md", "z.md"]


def test_ignores_invalid_utf8_bytes(tmp_path: Path) -> None:
    (tmp_path / "invalid.md").write_bytes(b"valid\xffcontent")

    documents = MarkdownLoader(tmp_path).load()

    assert documents[0].content == "validcontent"
