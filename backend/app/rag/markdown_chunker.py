import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.rag.markdown_loader import MarkdownDocument


_HEADER_PATTERN = re.compile(
    r"^[ \t]{0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*(?:\r?\n)?$"
)
_FENCE_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


@dataclass
class MarkdownChunk:
    content: str
    filename: str
    relative_path: str
    section: str | None
    chunk_index: int


class MarkdownChunker:
    def __init__(self, chunk_size: int = 3000, chunk_overlap: int = 300) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: MarkdownDocument) -> list[MarkdownChunk]:
        if not document.content.strip():
            return []

        chunks: list[MarkdownChunk] = []
        for section, section_content in self._split_sections(document.content):
            for content in self._split_large_text(section_content):
                chunks.append(
                    MarkdownChunk(
                        content=content,
                        filename=document.filename,
                        relative_path=document.relative_path,
                        section=section,
                        chunk_index=len(chunks),
                    )
                )

        return chunks

    def chunk_documents(
        self, documents: Iterable[MarkdownDocument]
    ) -> list[MarkdownChunk]:
        chunks: list[MarkdownChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def _split_sections(self, content: str) -> list[tuple[str | None, str]]:
        sections: list[tuple[str | None, str]] = []
        current_section: str | None = None
        current_lines: list[str] = []
        fence_marker: str | None = None

        for line in content.splitlines(keepends=True):
            fence_match = _FENCE_PATTERN.match(line)
            if fence_marker is not None:
                current_lines.append(line)
                if self._closes_fence(line, fence_marker):
                    fence_marker = None
                continue

            if fence_match:
                fence_marker = fence_match.group(1)
                current_lines.append(line)
                continue

            header_match = _HEADER_PATTERN.match(line)
            if not header_match:
                current_lines.append(line)
                continue

            self._append_section(sections, current_section, current_lines)
            current_section = self._clean_header_title(header_match.group(2))
            current_lines = [line]

        self._append_section(sections, current_section, current_lines)
        return sections

    @staticmethod
    def _append_section(
        sections: list[tuple[str | None, str]],
        section: str | None,
        lines: list[str],
    ) -> None:
        content = "".join(lines).strip()
        if content:
            sections.append((section, content))

    @staticmethod
    def _clean_header_title(title: str | None) -> str | None:
        if title is None:
            return None

        cleaned_title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
        return cleaned_title or None

    @staticmethod
    def _closes_fence(line: str, fence_marker: str) -> bool:
        match = _FENCE_PATTERN.match(line)
        if match is None:
            return False

        marker = match.group(1)
        remainder = line[match.end() :].strip()
        return (
            marker[0] == fence_marker[0]
            and len(marker) >= len(fence_marker)
            and not remainder
        )

    def _split_large_text(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            maximum_end = min(start + self.chunk_size, len(text))
            if maximum_end == len(text):
                end = maximum_end
            else:
                end = self._find_split_position(text, start, maximum_end)

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end == len(text):
                break

            next_start = self._find_overlap_start(text, start, end)
            start = next_start if next_start > start else end

        return chunks

    def _find_split_position(self, text: str, start: int, maximum_end: int) -> int:
        minimum_end = min(
            maximum_end,
            start + max(self.chunk_overlap + 1, self.chunk_size // 2),
        )

        for separator in ("\n\n", "\n", " "):
            position = text.rfind(separator, minimum_end, maximum_end)
            if position != -1:
                return position + len(separator)

        return maximum_end

    def _find_overlap_start(self, text: str, start: int, end: int) -> int:
        if self.chunk_overlap == 0:
            return end

        target = max(start + 1, end - self.chunk_overlap)
        for separator in ("\n\n", "\n", " "):
            position = text.find(separator, target, end)
            if position != -1 and position + len(separator) < end:
                return position + len(separator)

        for separator in ("\n\n", "\n", " "):
            position = text.rfind(separator, start + 1, target)
            if position != -1:
                return position + len(separator)

        return target
