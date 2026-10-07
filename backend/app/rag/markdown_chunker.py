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


@dataclass
class _MarkdownSection:
    section: str | None
    content: str
    useful_chars: int


class MarkdownChunker:
    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 300,
        min_chunk_chars: int = 80,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if min_chunk_chars <= 0:
            raise ValueError("min_chunk_chars must be greater than zero")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def chunk_document(self, document: MarkdownDocument) -> list[MarkdownChunk]:
        if not document.content.strip():
            return []

        sections = self._merge_small_sections(
            self._split_sections(document.content)
        )
        chunks: list[MarkdownChunk] = []
        for section in sections:
            for content in self._split_large_text(section.content):
                chunks.append(
                    MarkdownChunk(
                        content=content,
                        filename=document.filename,
                        relative_path=document.relative_path,
                        section=section.section,
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

    def _split_sections(self, content: str) -> list[_MarkdownSection]:
        sections: list[_MarkdownSection] = []
        current_section: str | None = None
        current_lines: list[str] = []
        heading_path: list[str | None] = [None] * 6
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
            level = len(header_match.group(1))
            title = self._clean_header_title(header_match.group(2))
            heading_path[level - 1] = title
            heading_path[level:] = [None] * (6 - level)
            hierarchy = [heading for heading in heading_path[:level] if heading]
            current_section = " > ".join(hierarchy) or None
            current_lines = [line]

        self._append_section(sections, current_section, current_lines)
        return sections

    def _append_section(
        self,
        sections: list[_MarkdownSection],
        section: str | None,
        lines: list[str],
    ) -> None:
        content = "".join(lines).strip()
        if content:
            sections.append(
                _MarkdownSection(
                    section=section,
                    content=content,
                    useful_chars=self._useful_content_length(content),
                )
            )

    def _merge_small_sections(
        self, sections: list[_MarkdownSection]
    ) -> list[_MarkdownSection]:
        merged: list[_MarkdownSection] = []
        pending: list[_MarkdownSection] = []

        for section in sections:
            if section.useful_chars < self.min_chunk_chars:
                pending.append(section)
                continue

            if pending:
                combined_content = self._join_section_content([*pending, section])
                merged.append(
                    _MarkdownSection(
                        section=section.section,
                        content=combined_content,
                        useful_chars=self._useful_content_length(combined_content),
                    )
                )
                pending = []
            else:
                merged.append(section)

        if pending:
            if merged:
                previous = merged.pop()
                combined_content = self._join_section_content([previous, *pending])
                merged.append(
                    _MarkdownSection(
                        section=previous.section,
                        content=combined_content,
                        useful_chars=self._useful_content_length(combined_content),
                    )
                )
            else:
                combined_content = self._join_section_content(pending)
                merged.append(
                    _MarkdownSection(
                        section=pending[-1].section,
                        content=combined_content,
                        useful_chars=self._useful_content_length(combined_content),
                    )
                )

        return merged

    @staticmethod
    def _join_section_content(sections: list[_MarkdownSection]) -> str:
        return "\n\n".join(section.content for section in sections)

    def _useful_content_length(self, content: str) -> int:
        useful_lines: list[str] = []
        fence_marker: str | None = None

        for line in content.splitlines():
            fence_match = _FENCE_PATTERN.match(line)
            if fence_marker is not None:
                useful_lines.append(line)
                if self._closes_fence(line, fence_marker):
                    fence_marker = None
                continue
            if fence_match:
                fence_marker = fence_match.group(1)
                useful_lines.append(line)
                continue
            if _HEADER_PATTERN.match(line):
                continue
            useful_lines.append(line)

        return len(" ".join("\n".join(useful_lines).split()))

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
        fenced_spans = self._fenced_code_spans(text)

        containing_span = self._containing_span(fenced_spans, maximum_end)
        if containing_span is not None:
            fence_start, fence_end = containing_span
            prefix_length = fence_start - start
            if (
                prefix_length < self.min_chunk_chars
                and fence_end - start <= self.chunk_size + self.min_chunk_chars
            ):
                return fence_end
            if prefix_length >= self.min_chunk_chars:
                return fence_start

        for separator in ("\n\n", "\n", " "):
            position = text.rfind(separator, minimum_end, maximum_end)
            while position != -1:
                split_position = position + len(separator)
                if self._containing_span(fenced_spans, split_position) is None:
                    return split_position
                position = text.rfind(separator, minimum_end, position)

        return maximum_end

    def _find_overlap_start(self, text: str, start: int, end: int) -> int:
        if self.chunk_overlap == 0:
            return end

        target = max(start + 1, end - self.chunk_overlap)
        fenced_spans = self._fenced_code_spans(text)
        containing_span = self._containing_span(fenced_spans, target)
        if containing_span is not None:
            fence_start, fence_end = containing_span
            if fence_start > start:
                return fence_start
            if fence_end < end:
                return fence_end

        for separator in ("\n\n", "\n", " "):
            position = text.find(separator, target, end)
            while position != -1:
                overlap_start = position + len(separator)
                if (
                    overlap_start < end
                    and self._containing_span(fenced_spans, overlap_start) is None
                ):
                    return overlap_start
                position = text.find(separator, position + len(separator), end)

        for separator in ("\n\n", "\n", " "):
            position = text.rfind(separator, start + 1, target)
            while position != -1:
                overlap_start = position + len(separator)
                if self._containing_span(fenced_spans, overlap_start) is None:
                    return overlap_start
                position = text.rfind(separator, start + 1, position)

        return target

    @staticmethod
    def _fenced_code_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        fence_marker: str | None = None
        fence_start = 0
        offset = 0

        for line in text.splitlines(keepends=True):
            match = _FENCE_PATTERN.match(line)
            if fence_marker is None and match:
                fence_marker = match.group(1)
                fence_start = offset
            elif fence_marker is not None and MarkdownChunker._closes_fence(
                line, fence_marker
            ):
                spans.append((fence_start, offset + len(line)))
                fence_marker = None
            offset += len(line)

        if fence_marker is not None:
            spans.append((fence_start, len(text)))
        return spans

    @staticmethod
    def _containing_span(
        spans: list[tuple[int, int]], position: int
    ) -> tuple[int, int] | None:
        for start, end in spans:
            if start < position < end:
                return start, end
        return None
