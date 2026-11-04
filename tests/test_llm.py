"""Клиент модели: разбор ответа, лестница режимов JSON, понятные ошибки. Сеть не нужна."""

import importlib
import json
from types import SimpleNamespace

import openai
import pytest

from kip_card.config import Settings
from kip_card.llm import LLMError, LLMResponseError, OpenAICompatibleLLM, extract_json

# openai<2 использует httpx, новые версии — httpx2; берём тот, что установлен вместе с SDK
try:
    httpx = importlib.import_module("httpx2")
except ImportError:  # pragma: no cover
    httpx = importlib.import_module("httpx")

REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
GOOD = {"seo_title": "t", "seo_description": "d", "search_queries": [], "commercial": {}, "data_gaps": []}


def status_error(cls, code: int, message: str = "error"):
    return cls(message, response=httpx.Response(code, request=REQUEST), body=None)


def completion(text: str, finish_reason: str = "stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)])


class FakeClient:
    def __init__(self, *results):
        self.results = list(results)
        self.kwargs: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def settings(**overrides) -> Settings:
    base = dict(base_url="https://openrouter.ai/api/v1", api_key="k", model="m", response_format="auto")
    return Settings(**{**base, **overrides})


@pytest.mark.parametrize("text", [
    json.dumps(GOOD),
    "```json\n" + json.dumps(GOOD) + "\n```",
    "Вот ответ:\n" + json.dumps(GOOD) + "\nГотово.",
])
def test_extract_json_variants(text):
    assert extract_json(text)["seo_title"] == "t"


@pytest.mark.parametrize("text", ["", "нет json", "[1, 2]", "{broken"])
def test_extract_json_errors(text):
    with pytest.raises(LLMResponseError):
        extract_json(text)


def test_json_schema_mode_and_openrouter_routing():
    client = FakeClient(completion(json.dumps(GOOD)))
    llm = OpenAICompatibleLLM(settings(), client=client)
    data, _ = llm.generate([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert data == GOOD
    sent = client.kwargs[0]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["extra_body"] == {"provider": {"require_parameters": True}}


def test_schema_retried_without_temperature():
    """Рассуждающие модели не принимают temperature — строгая схема должна сохраниться."""
    client = FakeClient(status_error(openai.NotFoundError, 404, "No endpoints found that support the requested parameters"),
                        completion(json.dumps(GOOD)), completion(json.dumps(GOOD)))
    llm = OpenAICompatibleLLM(settings(), client=client)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    llm.generate(messages)
    llm.generate(messages)
    assert llm.mode == "json_schema"
    assert "temperature" in client.kwargs[0]
    assert "temperature" not in client.kwargs[1] and "temperature" not in client.kwargs[2]
    assert client.kwargs[1]["response_format"]["type"] == "json_schema"


def test_fallback_to_json_object_when_schema_unsupported():
    client = FakeClient(status_error(openai.BadRequestError, 400, "response_format not supported"),
                        status_error(openai.BadRequestError, 400, "response_format not supported"),
                        completion(json.dumps(GOOD)))
    llm = OpenAICompatibleLLM(settings(), client=client)
    llm.generate([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert llm.mode == "json_object"
    assert client.kwargs[2]["response_format"] == {"type": "json_object"}
    assert "temperature" in client.kwargs[2]  # без схемы temperature снова отправляется
    assert "ФОРМАТ ОТВЕТА" in client.kwargs[2]["messages"][0]["content"]


def test_fallback_to_prompt_mode_for_local_server():
    client = FakeClient(status_error(openai.BadRequestError, 400), status_error(openai.BadRequestError, 400),
                        status_error(openai.BadRequestError, 400), completion(json.dumps(GOOD)))
    llm = OpenAICompatibleLLM(settings(base_url="http://localhost:11434/v1"), client=client)
    llm.generate([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert llm.mode == "prompt"
    assert "response_format" not in client.kwargs[3]
    assert "extra_body" not in client.kwargs[0]  # параметры OpenRouter не шлём локальному серверу


@pytest.mark.parametrize("error, expected", [
    (status_error(openai.AuthenticationError, 401), "API-ключ"),
    (status_error(openai.RateLimitError, 429), "лимит"),
    (status_error(openai.APIStatusError, 402), "Недостаточно средств"),
    (status_error(openai.InternalServerError, 503), "503"),
    (openai.APIConnectionError(request=REQUEST), "Не удалось подключиться"),
    (openai.APITimeoutError(request=REQUEST), "не ответила"),
])
def test_errors_are_human_readable(error, expected):
    llm = OpenAICompatibleLLM(settings(response_format="json_object"), client=FakeClient(error))
    with pytest.raises(LLMError, match=expected):
        llm.generate([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])


def test_truncated_answer_is_response_error():
    llm = OpenAICompatibleLLM(settings(), client=FakeClient(completion('{"seo_title": "t', "length")))
    with pytest.raises(LLMResponseError, match="обрезан"):
        llm.generate([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
