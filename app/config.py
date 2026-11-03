"""Настройки сервиса из переменных окружения.

Ключи API никогда не хранятся в коде: значения читаются из окружения или из
файла ``.env``, который добавлен в ``.gitignore``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class SettingsError(RuntimeError):
    """Некорректные настройки окружения."""


def load_dotenv(path: Path | str = ENV_FILE) -> None:
    """Подхватывает переменные из ``.env``, не перетирая уже заданные в окружении."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name}: ожидалось число, получено {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name}: ожидалось целое число, получено {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


@dataclass(frozen=True)
class Settings:
    """Параметры подключения к LLM и генерации."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    timeout: float = 90.0
    max_retries: int = 3
    temperature: float = 0.2
    json_mode: bool = True
    max_tokens: int = 4000

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        base_url = (os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise SettingsError(
                f"LLM_BASE_URL должен начинаться с http:// или https://, получено {base_url!r}"
            )
        return cls(
            base_url=base_url,
            api_key=(os.getenv("LLM_API_KEY") or "").strip(),
            model=(os.getenv("LLM_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            timeout=_env_float("LLM_TIMEOUT", 90.0),
            max_retries=max(0, _env_int("LLM_MAX_RETRIES", 3)),
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            json_mode=_env_bool("LLM_JSON_MODE", True),
            max_tokens=_env_int("LLM_MAX_TOKENS", 4000),
        )

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def is_configured(self) -> bool:
        """Настроен ли ключ (плейсхолдер из .env.example ключом не считается)."""
        return bool(self.api_key) and not self.api_key.startswith(("sk-...", "your", "<"))
