"""Тесты HTTP-слоя: валидация, форматы ответа, обработка ошибок."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm import LLMError, LLMResult  # noqa: E402
from app.main import create_app  # noqa: E402


class FakeClient:
    def __init__(self, draft: dict | None = None, error: Exception | None = None):
        self.draft = draft
        self.error = error

    async def complete_json(self, system: str, user: str, **kwargs):
        if self.error:
            raise self.error
        return self.draft, LLMResult(
            content="{}", model="fake-model", prompt_tokens=10, completion_tokens=5, latency_ms=5
        )


def make_app(draft: dict | None = None, error: Exception | None = None) -> TestClient:
    settings = Settings(base_url="https://llm.test/v1", api_key="test", model="test-model")
    app = create_app(settings, client_factory=lambda _: FakeClient(draft, error))
    return TestClient(app)


def test_health_reports_configuration():
    response = make_app().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True


def test_index_renders_form():
    response = make_app().get("/")
    assert response.status_code == 200
    assert "Данные товара" in response.text
    assert "Сгенерировать карточку" in response.text


def test_example_endpoint_returns_json():
    response = make_app().get("/api/example")
    assert response.status_code == 200
    assert "name" in response.json()


def test_invalid_json_body_returns_422():
    response = make_app().post(
        "/api/cards", content="{не json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_empty_name_returns_field_issue():
    response = make_app().post("/api/cards", json={"name": ""})
    assert response.status_code == 422
    issues = response.json()["error"]["details"]["issues"]
    assert issues[0]["field"] == "name"


def test_card_is_generated(good_draft, product_payload):
    response = make_app(good_draft).post("/api/cards?format=both", json=product_payload)
    assert response.status_code == 200

    body = response.json()
    assert body["card"]["seo_title"] == good_draft["seo_title"]
    assert body["card"]["fact_check"]["passed"] is True
    assert body["markdown"].startswith("# Манометр")


def test_markdown_format_returns_text(good_draft, product_payload):
    response = make_app(good_draft).post("/api/cards?format=markdown", json=product_payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")


def test_unknown_format_is_rejected(good_draft, product_payload):
    response = make_app(good_draft).post("/api/cards?format=xml", json=product_payload)
    assert response.status_code == 422
    assert response.json()["error"]["details"]["issues"][0]["field"] == "format"


def test_batch_reports_per_item_results(good_draft, product_payload):
    response = make_app(good_draft).post(
        "/api/cards/batch", json={"products": [product_payload, {"name": ""}]}
    )
    assert response.status_code == 200

    results = response.json()["results"]
    assert results[0]["card"]["seo_title"]
    assert results[1]["error"]["code"] == "invalid_input"


def test_batch_limit_is_enforced(product_payload):
    response = make_app().post("/api/cards/batch", json={"products": [product_payload] * 11})
    assert response.status_code == 422


def test_llm_error_is_mapped_to_http_status(product_payload):
    client = make_app(error=LLMError("Ключ отклонён", code="llm_auth", status=401))
    response = client.post("/api/cards", json=product_payload)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_auth"


def test_missing_key_is_reported_as_503(product_payload):
    settings = Settings(base_url="https://llm.test/v1", api_key="", model="test-model")
    client = TestClient(create_app(settings))

    response = client.post("/api/cards", json=product_payload)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"


@pytest.mark.parametrize("payload", ["строка", 42, [], None])
def test_non_object_payload_is_rejected(payload):
    response = make_app().post("/api/cards", json=payload)
    assert response.status_code == 422
