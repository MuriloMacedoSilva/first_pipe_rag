from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.rag import llm
from app.rag.llm import GoogleLLMService, LLMError


def mock_client(monkeypatch: pytest.MonkeyPatch, response: object) -> Mock:
    generate_content = Mock(return_value=response)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(llm.genai, "Client", Mock(return_value=client))
    return generate_content


def test_missing_api_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client_factory = Mock()
    monkeypatch.setattr(llm, "GOOGLE_API_KEY", None)
    monkeypatch.setattr(llm.genai, "Client", client_factory)

    with pytest.raises(LLMError, match="Google API key is required"):
        GoogleLLMService()

    client_factory.assert_not_called()


def test_generate_rejects_empty_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_content = mock_client(monkeypatch, SimpleNamespace(text="Answer"))
    service = GoogleLLMService(api_key="test-key")

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        service.generate(" \n\t")

    generate_content.assert_not_called()


def test_generate_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_content = mock_client(monkeypatch, SimpleNamespace(text="Answer"))

    GoogleLLMService(api_key="test-key", model="custom-model").generate("Prompt")

    assert generate_content.call_args.kwargs["model"] == "custom-model"


def test_generate_sends_prompt_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_content = mock_client(monkeypatch, SimpleNamespace(text="Answer"))

    GoogleLLMService(api_key="test-key").generate("Expected prompt")

    assert generate_content.call_args.kwargs["contents"] == "Expected prompt"


def test_generate_returns_trimmed_string(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client(monkeypatch, SimpleNamespace(text=123))

    result = GoogleLLMService(api_key="test-key").generate("Prompt")

    assert result == "123"
    assert isinstance(result, str)


def test_generate_rejects_empty_api_response(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client(monkeypatch, SimpleNamespace(text="   "))

    with pytest.raises(LLMError, match="empty text"):
        GoogleLLMService(api_key="test-key").generate("Prompt")


def test_api_key_is_redacted_from_external_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-test-key"
    generate_content = mock_client(monkeypatch, SimpleNamespace(text="Answer"))
    generate_content.side_effect = RuntimeError(f"request failed using {secret}")

    with pytest.raises(LLMError) as error_info:
        GoogleLLMService(api_key=secret).generate("Prompt")

    assert secret not in str(error_info.value)
    assert "[REDACTED]" in str(error_info.value)
