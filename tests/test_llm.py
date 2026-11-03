"""Тесты клиента LLM: повторы, коды ошибок, разбор JSON (без сети)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.llm import LLMClient, LLMError, extract_json


def settings(**overrides) -> Settings:
    base = dict(
        base_url="https://llm.test/v1",
        api_key="test-key",
        model="test-model",
        timeout=5.0,
        max_retries=2,
        temperature=0.0,
        json_mode=True,
    )
    base.update(overrides)
    return Settings(**base)


def run(coro):
    return asyncio.run(coro)


def client_with(handler, **overrides) -> LLMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return LLMClient(settings(**overrides), client=http, sleep=lambda _: asyncio.sleep(0))


def ok_response(content: str = '{"ok": true}') -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "test-model",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def test_successful_call_returns_content_and_usage():
    client = client_with(lambda request: ok_response())
    result = run(client.complete("system", "user"))

    assert result.content == '{"ok": true}'
    assert result.total_tokens == 18
    assert result.attempts == 1


def test_retry_on_server_error_then_success():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return ok_response()

    result = run(client_with(handler).complete("system", "user"))

    assert calls["count"] == 2
    assert result.attempts == 2


def test_auth_error_is_not_retried():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(LLMError) as error:
        run(client_with(handler).complete("system", "user"))

    assert error.value.code == "llm_auth"
    assert calls["count"] == 1


def test_rate_limit_message_with_retry_after_is_retried():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                402,
                json={"error": {"message": "Free limit reached: Retry after 0s"}},
            )
        return ok_response()

    result = run(client_with(handler).complete("system", "user"))

    assert calls["count"] == 2
    assert result.attempts == 2


def test_timeout_is_mapped_to_llm_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow")

    with pytest.raises(LLMError) as error:
        run(client_with(handler, max_retries=0).complete("system", "user"))

    assert error.value.code == "llm_timeout"
    assert error.value.retryable is True


def test_missing_key_is_reported_before_request():
    client = LLMClient(settings(api_key=""))
    with pytest.raises(LLMError) as error:
        run(client.complete("system", "user"))
    assert error.value.code == "llm_not_configured"


def test_json_mode_falls_back_when_provider_rejects_response_format():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "response_format" in payload:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format is not supported"}},
            )
        return ok_response()

    client = client_with(handler)
    data, _ = run(client.complete_json("system", "user"))

    assert data == {"ok": True}
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]


def test_invalid_json_triggers_single_repair_request():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return ok_response("Вот карточка: не совсем JSON")
        return ok_response('```json\n{"seo_title": "Манометр"}\n```')

    data, result = run(client_with(handler).complete_json("system", "user"))

    assert calls["count"] == 2
    assert data["seo_title"] == "Манометр"
    assert result.content.startswith("```json")


def test_extra_body_is_sent_to_provider():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return ok_response()

    client = client_with(handler, extra_body={"reasoning_effort": "low"})
    run(client.complete("system", "user"))

    assert payloads[0]["reasoning_effort"] == "low"


def test_empty_response_is_an_error():
    client = client_with(lambda request: ok_response(""))
    with pytest.raises(LLMError) as error:
        run(client.complete("system", "user"))
    assert error.value.code == "llm_empty_response"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Текст до {"a": {"b": "}"}} текст после', {"a": {"b": "}"}}),
        ('[1, 2]', None),
    ],
)
def test_extract_json(raw: str, expected: dict | None):
    if expected is None:
        with pytest.raises(LLMError):
            extract_json(raw)
        return
    assert extract_json(raw) == expected
