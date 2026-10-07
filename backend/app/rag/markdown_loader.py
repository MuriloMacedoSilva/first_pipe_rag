from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarkdownDocument:
    content: str
    filename: str
    relative_path: str
    absolute_path: str


class MarkdownLoader:
    def __init__(self, material_path: str | Path) -> None:
        self.material_path = Path(material_path).expanduser().resolve()

    def load(self) -> list[MarkdownDocument]:
        if not self.material_path.exists():
            raise FileNotFoundError(
                f"Material directory does not exist: {self.material_path}"
            )

        if not self.material_path.is_dir():
            raise NotADirectoryError(
                f"Material path is not a directory: {self.material_path}"
            )

        markdown_files = sorted(
            self.material_path.rglob("*.md"),
            key=lambda path: path.relative_to(self.material_path).as_posix(),
        )

        return [
            MarkdownDocument(
                content=file_path.read_text(encoding="utf-8", errors="ignore"),
                filename=file_path.name,
                relative_path=file_path.relative_to(self.material_path).as_posix(),
                absolute_path=str(file_path.resolve()),
            )
            for file_path in markdown_files
        ]
