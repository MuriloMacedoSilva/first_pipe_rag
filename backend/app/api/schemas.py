from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be empty")
        return value


class SourceResponse(BaseModel):
    filename: str
    relative_path: str
    section: str | None
    chunk_index: int
    distance: float | None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
