import pytest

from app.rag.markdown_chunker import MarkdownChunker
from app.rag.markdown_loader import MarkdownDocument


def make_document(
    content: str,
    filename: str = "document.md",
    relative_path: str = "docs/document.md",
) -> MarkdownDocument:
    return MarkdownDocument(
        content=content,
        filename=filename,
        relative_path=relative_path,
        absolute_path=f"/material/{relative_path}",
    )


def test_small_document_generates_single_chunk() -> None:
    chunks = MarkdownChunker().chunk_document(make_document("# Title\n\nContent"))

    assert len(chunks) == 1
    assert chunks[0].content == "# Title\n\nContent"


def test_empty_document_generates_no_chunks() -> None:
    chunker = MarkdownChunker()

    assert chunker.chunk_document(make_document("")) == []
    assert chunker.chunk_document(make_document(" \n\t")) == []


def test_document_without_headers_is_supported() -> None:
    chunks = MarkdownChunker().chunk_document(
        make_document("First paragraph.\n\nSecond paragraph.")
    )

    assert len(chunks) == 1
    assert chunks[0].section is None
    assert "Second paragraph." in chunks[0].content


def test_document_is_split_by_markdown_headers() -> None:
    content = "# Introduction\nIntro text.\n\n## Install\nInstall text."

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert [chunk.content for chunk in chunks] == [
        "# Introduction\nIntro text.",
        "## Install\nInstall text.",
    ]


def test_section_metadata_uses_header_title() -> None:
    content = "## Installation\nInstructions.\n\n### Configuration ###\nSettings."

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert [chunk.section for chunk in chunks] == ["Installation", "Configuration"]


def test_filename_is_preserved() -> None:
    chunks = MarkdownChunker().chunk_document(
        make_document("Content", filename="spring.md")
    )

    assert chunks[0].filename == "spring.md"


def test_relative_path_is_preserved() -> None:
    chunks = MarkdownChunker().chunk_document(
        make_document("Content", relative_path="java/spring.md")
    )

    assert chunks[0].relative_path == "java/spring.md"


def test_chunk_index_starts_at_zero() -> None:
    content = "# First\nContent.\n\n## Second\nMore content."

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_multiple_documents_restart_chunk_index() -> None:
    documents = [
        make_document("# One\nText.\n\n## Two\nText.", filename="a.md"),
        make_document("# Three\nText.\n\n## Four\nText.", filename="b.md"),
    ]

    chunks = MarkdownChunker().chunk_documents(documents)

    assert [chunk.chunk_index for chunk in chunks if chunk.filename == "a.md"] == [0, 1]
    assert [chunk.chunk_index for chunk in chunks if chunk.filename == "b.md"] == [0, 1]


def test_large_section_is_subdivided() -> None:
    content = "# Large\n\n" + " ".join(f"word{i}" for i in range(80))

    chunks = MarkdownChunker(chunk_size=100, chunk_overlap=20).chunk_document(
        make_document(content)
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert all(chunk.section == "Large" for chunk in chunks)


def test_large_section_subdivisions_have_overlap() -> None:
    content = " ".join(f"word{i}" for i in range(60))

    chunks = MarkdownChunker(chunk_size=80, chunk_overlap=25).chunk_document(
        make_document(content)
    )

    for previous, current in zip(chunks, chunks[1:]):
        assert set(previous.content.split()) & set(current.content.split())


def test_chunks_preserve_original_section_order() -> None:
    content = "# Alpha\nA.\n\n## Beta\nB.\n\n### Gamma\nC."

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert [chunk.section for chunk in chunks] == ["Alpha", "Beta", "Gamma"]


def test_invalid_configurations_raise_value_error() -> None:
    invalid_configurations = [
        {"chunk_size": 0, "chunk_overlap": 0},
        {"chunk_size": 100, "chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 101},
    ]

    for configuration in invalid_configurations:
        with pytest.raises(ValueError):
            MarkdownChunker(**configuration)


def test_headers_from_level_one_through_six_are_recognized() -> None:
    content = "\n\n".join(
        f"{'#' * level} Level {level}\nContent {level}." for level in range(1, 7)
    )

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert [chunk.section for chunk in chunks] == [
        "Level 1",
        "Level 2",
        "Level 3",
        "Level 4",
        "Level 5",
        "Level 6",
    ]


def test_large_text_prefers_paragraph_boundaries() -> None:
    first = "A" * 30
    second = "B" * 30
    third = "C" * 30
    content = f"{first}\n\n{second}\n\n{third}"

    chunks = MarkdownChunker(chunk_size=70, chunk_overlap=5).chunk_document(
        make_document(content)
    )

    assert chunks[0].content == f"{first}\n\n{second}"


def test_header_like_lines_inside_fenced_code_are_not_sections() -> None:
    content = "# Examples\n\n```markdown\n## Not a section\n```\n\nAfter code."

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert len(chunks) == 1
    assert chunks[0].section == "Examples"
    assert "## Not a section" in chunks[0].content
