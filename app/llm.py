"""Асинхронный клиент к OpenAI-совместимому Chat Completions API.

Клиент не зависит от конкретного вендора: OpenAI, OpenRouter, DeepSeek, локальные
Ollama/vLLM — достаточно поменять ``LLM_BASE_URL`` и ``LLM_MODEL``.

Что здесь обрабатывается:
* временные ошибки провайдера (429/5xx/timeout) — повтор с экспоненциальной паузой
  и учётом заголовка ``Retry-After``;
* невалидный JSON в ответе — один автоматический запрос на исправление;
* внятные сообщения об ошибках (ключ, лимит, недоступность, таймаут), чтобы
  FastAPI мог вернуть понятный статус, а не 500.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from app.config import Settings

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 8.0
MAX_RETRY_AFTER_SECONDS = 90.0


class LLMError(RuntimeError):
    """Ошибка обращения к LLM с машинным кодом для API-слоя."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "llm_error",
        status: int | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


@dataclass(frozen=True)
class LLMResult:
    """Успешный ответ модели."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_meta(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
        }


class LLMClient:
    """Тонкая обёртка над ``POST /chat/completions``."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self._client = client
        self._sleep = sleep
        self._json_mode_supported = settings.json_mode

    # ------------------------------------------------------------------ public

    async def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Отправляет пару сообщений и возвращает текстовый ответ модели."""
        if not self.settings.is_configured:
            raise LLMError(
                "Не задан LLM_API_KEY: скопируйте .env.example в .env и укажите ключ API",
                code="llm_not_configured",
            )

        use_json = self._json_mode_supported if json_mode is None else json_mode and self._json_mode_supported
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error: LLMError | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                result = await self._request(
                    messages,
                    json_mode=use_json,
                    temperature=self.settings.temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.settings.max_tokens,
                    attempt=attempt,
                )
                return result
            except LLMError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.settings.max_retries:
                    raise
                await self._sleep(self._backoff(attempt, exc))

        raise last_error or LLMError("LLM недоступна", code="llm_unreachable")

    async def complete_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], LLMResult]:
        """Возвращает разобранный JSON и метаданные ответа.

        Если модель вернула невалидный JSON, делается один запрос на исправление:
        это дешевле и надёжнее, чем показывать пользователю ошибку парсинга.
        """
        result = await self.complete(system, user, temperature=temperature, max_tokens=max_tokens)
        try:
            return extract_json(result.content), result
        except LLMError as exc:
            if exc.code != "llm_invalid_json":
                raise
            repair_prompt = (
                f"{user}\n\n---\nПредыдущий ответ был невалидным JSON:\n"
                f"{result.content[:3000]}\n\n"
                "Верни ТОЛЬКО валидный JSON-объект по требуемой схеме. "
                "Без markdown, без комментариев, без текста до и после JSON."
            )
            repaired = await self.complete(system, repair_prompt, temperature=temperature, max_tokens=max_tokens)
            return extract_json(repaired.content), repaired

    # ----------------------------------------------------------------- private

    async def _request(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        attempt: int,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.settings.extra_body:
            payload.update(self.settings.extra_body)

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        started = time.perf_counter()
        try:
            response = await self._post(payload, headers)
            if response.status_code == 400 and json_mode:
                text = response.text.lower()
                if "response_format" in text or "json_object" in text:
                    # Провайдер не поддерживает JSON-режим — повторяем без него и запоминаем.
                    self._json_mode_supported = False
                    payload.pop("response_format", None)
                    response = await self._post(payload, headers)
            latency_ms = int((time.perf_counter() - started) * 1000)
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"LLM не ответила за {self.settings.timeout:.0f} c",
                code="llm_timeout",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Не удалось подключиться к LLM по адресу {self.settings.base_url}: {exc}",
                code="llm_unreachable",
                retryable=True,
            ) from exc

        if response.status_code != 200:
            raise _status_error(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(
                "LLM вернула ответ, который не является JSON",
                code="llm_bad_response",
                status=response.status_code,
            ) from exc

        content = _message_content(data)
        if not content.strip():
            raise LLMError(
                "LLM вернула пустой ответ",
                code="llm_empty_response",
                retryable=True,
                details={"model": data.get("model")},
            )

        usage = data.get("usage") or {}
        return LLMResult(
            content=content,
            model=str(data.get("model") or self.settings.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            attempts=attempt + 1,
        )

    async def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        if self._client is not None:
            return await self._client.post(
                self.settings.chat_completions_url,
                json=payload,
                headers=headers,
                timeout=self.settings.timeout,
            )
        async with httpx.AsyncClient(timeout=self.settings.timeout) as client:
            return await client.post(
                self.settings.chat_completions_url, json=payload, headers=headers
            )

    @staticmethod
    def _backoff(attempt: int, error: LLMError) -> float:
        retry_after = error.details.get("retry_after")
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            return min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
        base = min(MAX_BACKOFF_SECONDS, 0.8 * (2**attempt))
        return base * (0.7 + random.random() * 0.6)


# ---------------------------------------------------------------------- helpers


def _with_attempts(result: LLMResult, attempts: int) -> LLMResult:
    return LLMResult(
        content=result.content,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
        attempts=attempts,
    )


def _message_content(data: dict[str, Any]) -> str:
    """Достаёт текст ответа: строка или массив частей (мультимодальные ответы)."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {None, "text"}
        ]
        return "\n".join(part for part in parts if part)
    return ""


def _status_error(response: httpx.Response) -> LLMError:
    """Превращает HTTP-ошибку провайдера в понятное сообщение."""
    status = response.status_code
    detail = _error_message(response)
    details: dict[str, Any] = {"status": status}
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            details["retry_after"] = float(retry_after)
        except ValueError:
            pass

    if status in {401, 403}:
        return LLMError(
            "Ключ API отклонён провайдером. Проверьте LLM_API_KEY и права доступа.",
            code="llm_auth",
            status=status,
            details=details,
        )
    if status == 404:
        return LLMError(
            f"Модель или эндпоинт не найдены ({detail or '404'}). Проверьте LLM_BASE_URL и LLM_MODEL.",
            code="llm_not_found",
            status=status,
            details=details,
        )
    if status == 429:
        return LLMError(
            "Превышен лимит запросов к LLM. Повторите попытку позже.",
            code="llm_rate_limit",
            status=status,
            retryable=True,
            details=details,
        )
    if status == 402:
        # Некоторые бесплатные шлюзы используют 402 для лимита «N запросов в минуту».
        retry_after = _retry_after_from_message(detail)
        if retry_after is not None:
            details["retry_after"] = retry_after
            return LLMError(
                f"Лимит бесплатного доступа к LLM: повтор через {retry_after:.0f} c.",
                code="llm_rate_limit",
                status=status,
                retryable=True,
                details=details,
            )
        return LLMError(
            f"Платёжный лимит провайдера LLM (402): {detail or 'требуется оплата'}",
            code="llm_payment_required",
            status=status,
            details=details,
        )
    if status in RETRYABLE_STATUS:
        return LLMError(
            f"Провайдер LLM вернул временную ошибку {status}. Попробуйте ещё раз.",
            code="llm_unavailable",
            status=status,
            retryable=True,
            details=details,
        )
    return LLMError(
        f"LLM отклонила запрос ({status}): {detail or 'без описания'}",
        code="llm_bad_request",
        status=status,
        details=details,
    )


def _retry_after_from_message(message: str) -> float | None:
    """Достаёт паузу из текста вида «Retry after 38s»."""
    match = re.search(r"retry\s+after\s+(\d+(?:[.,]\d+)?)\s*s", message, re.IGNORECASE)
    if not match:
        return None
    return min(float(match.group(1).replace(",", ".")), MAX_RETRY_AFTER_SECONDS)


def _error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        if isinstance(error, str):
            return error[:300]
        if "message" in data:
            return str(data["message"])[:300]
    return json.dumps(data, ensure_ascii=False)[:300]


def extract_json(text: str) -> dict[str, Any]:
    """Извлекает JSON-объект из ответа модели.

    Модели любят оборачивать JSON в ```json ... ``` или добавлять текст.
    Порядок попыток: как есть → снять markdown-обёртку → найти сбалансированный объект.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMError("Пустой ответ модели — разбирать нечего", code="llm_invalid_json")

    candidates: list[str] = [text.strip()]

    stripped = text.strip()
    if stripped.startswith("```"):
        without_fences = stripped.strip("`")
        if without_fences.lower().startswith("json"):
            without_fences = without_fences[4:]
        candidates.append(without_fences.strip())

    balanced = _balanced_object(text)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise LLMError(
        "Модель вернула невалидный JSON",
        code="llm_invalid_json",
        details={"snippet": text[:300]},
    )


def _balanced_object(text: str) -> str | None:
    """Возвращает первый сбалансированный JSON-объект из текста."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None
