import pytest

from app.rag.markdown_chunker import MarkdownChunk, MarkdownChunker
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


def long_text(label: str) -> str:
    return " ".join(f"{label}{index}" for index in range(20))


def reconstruct_words(chunks: list[MarkdownChunk]) -> list[str]:
    reconstructed = chunks[0].content.split()
    for chunk in chunks[1:]:
        words = chunk.content.split()
        overlap = 0
        for size in range(min(len(reconstructed), len(words)), 0, -1):
            if reconstructed[-size:] == words[:size]:
                overlap = size
                break
        reconstructed.extend(words[overlap:])
    return reconstructed


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


def test_isolated_header_is_merged_with_following_content() -> None:
    content = (
        "# Neural Networks\n\n"
        "## Activation Functions\n\n"
        "### ReLU\n\n"
        f"{long_text('relu')}"
    )

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert len(chunks) == 1
    assert "# Neural Networks" in chunks[0].content
    assert "## Activation Functions" in chunks[0].content
    assert "### ReLU" in chunks[0].content
    assert chunks[0].content.strip() != "## Activation Functions"


def test_short_section_is_merged_with_neighboring_section() -> None:
    content = (
        "## Introduction\n\nBrief introduction.\n\n"
        f"## Concept\n\n{long_text('concept')}"
    )

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert len(chunks) == 1
    assert "## Introduction" in chunks[0].content
    assert "## Concept" in chunks[0].content


def test_substantial_sections_remain_separate() -> None:
    content = (
        f"# Introduction\n\n{long_text('intro')}\n\n"
        f"## Installation\n\n{long_text('install')}"
    )

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert len(chunks) == 2
    assert chunks[0].section == "Introduction"
    assert chunks[1].section == "Introduction > Installation"


def test_header_hierarchy_is_preserved_in_section_metadata() -> None:
    content = (
        "# Neural Networks\n\n"
        "## Activation Functions\n\n"
        f"### ReLU\n\n{long_text('activation')}"
    )

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert chunks[0].section == "Neural Networks > Activation Functions > ReLU"


def test_large_section_is_subdivided() -> None:
    content = "# Large\n\n" + " ".join(f"word{index}" for index in range(80))

    chunks = MarkdownChunker(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_chars=20,
    ).chunk_document(make_document(content))

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 100 for chunk in chunks)
    assert all(chunk.section == "Large" for chunk in chunks)


def test_large_section_subdivisions_have_overlap() -> None:
    content = " ".join(f"word{index}" for index in range(60))

    chunks = MarkdownChunker(
        chunk_size=80,
        chunk_overlap=25,
        min_chunk_chars=20,
    ).chunk_document(make_document(content))

    for previous, current in zip(chunks, chunks[1:]):
        assert set(previous.content.split()) & set(current.content.split())


def test_complete_content_is_preserved_when_removing_overlap() -> None:
    content = " ".join(f"token{index}" for index in range(100))
    chunks = MarkdownChunker(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_chars=20,
    ).chunk_document(make_document(content))

    assert reconstruct_words(chunks) == content.split()


def test_original_section_order_is_preserved() -> None:
    content = (
        f"# Alpha\n\n{long_text('alpha')}\n\n"
        f"## Beta\n\n{long_text('beta')}\n\n"
        f"### Gamma\n\n{long_text('gamma')}"
    )

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert [chunk.section for chunk in chunks] == [
        "Alpha",
        "Alpha > Beta",
        "Alpha > Beta > Gamma",
    ]
    combined = "\n".join(chunk.content for chunk in chunks)
    assert combined.index("# Alpha") < combined.index("## Beta")
    assert combined.index("## Beta") < combined.index("### Gamma")


def test_fenced_code_block_is_not_split_when_it_fits() -> None:
    code = (
        "```python\n"
        "def backward(loss, weights):\n"
        "    gradient = loss.gradient(weights)\n"
        "    return weights - gradient\n"
        "```"
    )
    content = f"# Example\n\n{'A' * 85}\n\n{code}\n\n{'B' * 50}"

    chunks = MarkdownChunker(
        chunk_size=130,
        chunk_overlap=10,
        min_chunk_chars=30,
    ).chunk_document(make_document(content))

    assert any(code in chunk.content for chunk in chunks)
    for chunk in chunks:
        if "```python" in chunk.content:
            assert chunk.content.count("```") == 2


def test_header_like_lines_inside_fenced_code_are_not_sections() -> None:
    content = (
        "# Examples\n\n"
        "```markdown\n## Not a section\n```\n\n"
        f"{long_text('example')}"
    )

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert len(chunks) == 1
    assert chunks[0].section == "Examples"
    assert "## Not a section" in chunks[0].content


def test_chunk_indexes_are_sequential() -> None:
    content = "# Large\n\n" + " ".join(f"word{index}" for index in range(80))

    chunks = MarkdownChunker(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_chars=20,
    ).chunk_document(make_document(content))

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_multiple_documents_restart_chunk_index() -> None:
    documents = [
        make_document(long_text("first"), filename="a.md"),
        make_document(long_text("second"), filename="b.md"),
    ]

    chunks = MarkdownChunker(
        chunk_size=80,
        chunk_overlap=10,
        min_chunk_chars=20,
    ).chunk_documents(documents)

    assert [chunk.chunk_index for chunk in chunks if chunk.filename == "a.md"] == list(
        range(sum(chunk.filename == "a.md" for chunk in chunks))
    )
    assert [chunk.chunk_index for chunk in chunks if chunk.filename == "b.md"] == list(
        range(sum(chunk.filename == "b.md" for chunk in chunks))
    )


def test_chunking_is_deterministic() -> None:
    document = make_document(
        "# Guide\n\n"
        "## Brief\n\nShort.\n\n"
        f"## Details\n\n{long_text('detail')}"
    )
    chunker = MarkdownChunker(chunk_size=120, chunk_overlap=20)

    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)

    assert first == second


def test_very_small_sections_do_not_remain_isolated_when_mergeable() -> None:
    content = "## One\n\nA.\n\n## Two\n\nB.\n\n## Three\n\nC."

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert len(chunks) == 1
    assert all(header in chunks[0].content for header in ("## One", "## Two", "## Three"))


def test_filename_and_relative_path_are_preserved() -> None:
    chunks = MarkdownChunker().chunk_document(
        make_document("Content", filename="spring.md", relative_path="java/spring.md")
    )

    assert chunks[0].filename == "spring.md"
    assert chunks[0].relative_path == "java/spring.md"


def test_headers_from_level_one_through_six_build_hierarchy() -> None:
    headers = "\n\n".join(
        f"{'#' * level} Level {level}" for level in range(1, 7)
    )
    content = f"{headers}\n\n{long_text('deep')}"

    chunks = MarkdownChunker().chunk_document(make_document(content))

    assert chunks[0].section == " > ".join(
        f"Level {level}" for level in range(1, 7)
    )


def test_large_text_prefers_paragraph_boundaries() -> None:
    first = "A" * 30
    second = "B" * 30
    third = "C" * 30
    content = f"{first}\n\n{second}\n\n{third}"

    chunks = MarkdownChunker(
        chunk_size=70,
        chunk_overlap=5,
        min_chunk_chars=20,
    ).chunk_document(make_document(content))

    assert chunks[0].content == f"{first}\n\n{second}"


def test_invalid_configurations_raise_value_error() -> None:
    invalid_configurations = [
        {"chunk_size": 0, "chunk_overlap": 0},
        {"chunk_size": 100, "chunk_overlap": -1},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"chunk_size": 100, "chunk_overlap": 101},
        {"chunk_size": 100, "chunk_overlap": 10, "min_chunk_chars": 0},
    ]

    for configuration in invalid_configurations:
        with pytest.raises(ValueError):
            MarkdownChunker(**configuration)
