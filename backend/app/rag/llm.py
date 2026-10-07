from google import genai
from google.genai import types

from app.config import GOOGLE_API_KEY, GOOGLE_CHAT_MODEL


class LLMError(RuntimeError):
    pass


class GoogleLLMService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_api_key = api_key if api_key is not None else GOOGLE_API_KEY
        if not resolved_api_key or not resolved_api_key.strip():
            raise LLMError("Google API key is required for text generation")

        resolved_model = model if model is not None else GOOGLE_CHAT_MODEL
        if not resolved_model or not resolved_model.strip():
            raise LLMError("Google chat model is required")

        self._api_key = resolved_api_key.strip()
        self.model = resolved_model.strip()
        self._client = genai.Client(api_key=self._api_key)

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Generation prompt cannot be empty")

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                ),
            )
        except Exception as error:
            self._raise_request_error(error)

        text = getattr(response, "text", None)
        if text is None:
            raise LLMError("Google generation response did not contain text")

        result = str(text).strip()
        if not result:
            raise LLMError("Google generation response contained empty text")
        return result

    def _raise_request_error(self, error: Exception) -> None:
        details = str(error).replace(self._api_key, "[REDACTED]")
        message = f"Google generation request failed ({type(error).__name__})"
        if details:
            message = f"{message}: {details}"
        raise LLMError(message) from None
