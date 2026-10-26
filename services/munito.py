"""
Munito AI Service — Grok-style LLM and Vision assistant via OpenAI-compatible API
with per-chat SQLite configuration, model selection, and token usage tracking.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, witty, and concise AI assistant in a Telegram group chat. "
    "Format your answers clearly and respond in the language of the user's prompt (Ukrainian by default). "
    "If analyzing an image or text from chat, be accurate and informative."
)

DEFAULT_MODEL = "deepseek-ai/deepseek-v4.1-flash-1017"

AVAILABLE_MODELS = [
    "deepseek-ai/deepseek-v4.1-flash-1017",
    "qwen/qwen3.8-27b",
    "deepseek-ai/deepseek-v4-flash",
    "zai-org/glm-5.3-flash",
    "google/gemma-4-12b",
]


def resolve_default_ai_db() -> str:
    """Resolve the default SQLite path for AI assistant, falling back to local dir if data/ is not writable."""
    env_path = os.getenv("AI_DB")
    if env_path:
        return env_path
    data_dir = Path("data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".write_test_ai"
        test_file.touch()
        test_file.unlink()
        return "data/ai.db"
    except (PermissionError, OSError):
        return "ai.db"


@dataclass
class AIChatConfig:
    chat_id: int
    enabled: bool = True
    model: str = DEFAULT_MODEL
    system_prompt: str | None = None


@dataclass(frozen=True)
class AIChatStats:
    chat_id: int
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def get_next_model(current_model: str) -> str:
    """Cycle to the next available model."""
    try:
        idx = AVAILABLE_MODELS.index(current_model)
        return AVAILABLE_MODELS[(idx + 1) % len(AVAILABLE_MODELS)]
    except ValueError:
        return AVAILABLE_MODELS[0]


def format_ai_card(
    chat_title: str,
    chat_id: int,
    cfg: AIChatConfig,
    stats: AIChatStats,
) -> str:
    """Format AI assistant status, model, and token usage into an HTML message."""
    status_str = "🟢 Увімкнено (Enabled)" if cfg.enabled else "🔴 Вимкнено (Disabled)"
    return (
        f"🤖 <b>Налаштування Munito AI Асистента</b>\n"
        f"<b>Чат:</b> <code>{chat_title}</code> (<code>{chat_id}</code>)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ <b>Статус:</b> {status_str}\n"
        f"🧠 <b>Модель:</b> <code>{cfg.model}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Використання токенів:</b>\n"
        f"  • Всього запитів: <b>{stats.request_count:,}</b>\n"
        f"  • Промпт токенів: <b>{stats.prompt_tokens:,}</b>\n"
        f"  • Згенеровано токенів: <b>{stats.completion_tokens:,}</b>\n"
        f"  • Загалом токенів: <b>{stats.total_tokens:,}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Викликається виключно через тег бота: @bot &lt;запит&gt;</i>"
    )


def build_ai_keyboard(
    chat_id: int, cfg: AIChatConfig, confirm_reset: bool = False
) -> InlineKeyboardMarkup:
    """Construct inline keyboard for AI configuration."""
    toggle_text = "🔴 Вимкнути" if cfg.enabled else "🟢 Увімкнути"
    model_short = cfg.model.split("/")[-1]
    rows = [
        [
            InlineKeyboardButton(
                text=f"Статус: {toggle_text}",
                callback_data=f"ai:toggle:{chat_id}",
            )
        ],
        [
            InlineKeyboardButton(
                text=f"🧠 Модель: {model_short}",
                callback_data=f"ai:model:{chat_id}",
            )
        ],
    ]

    if confirm_reset:
        rows.append(
            [
                InlineKeyboardButton(
                    text="⚠️ Підтвердити скидання",
                    callback_data=f"ai:reset_confirm:{chat_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Скасувати",
                    callback_data=f"ai:refresh:{chat_id}",
                ),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🧹 Скинути лічильник токенів",
                    callback_data=f"ai:reset_prompt:{chat_id}",
                )
            ]
        )

    rows.append(
        [
            InlineKeyboardButton(
                text="🔄 Оновити",
                callback_data=f"ai:refresh:{chat_id}",
            )
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=rows)


def split_telegram_message(text: str, max_chars: int = 4000) -> list[str]:
    """
    Split a long response into chunks suitable for Telegram message limits (4096 chars).
    Prefers splitting by double newlines, single newlines, or spaces.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = text

    while len(current) > max_chars:
        split_at = current.rfind("\n\n", 0, max_chars)
        if split_at == -1 or split_at < max_chars // 2:
            split_at = current.rfind("\n", 0, max_chars)
        if split_at == -1 or split_at < max_chars // 2:
            split_at = current.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars

        chunks.append(current[:split_at].strip())
        current = current[split_at:].strip()

    if current:
        chunks.append(current)

    return chunks


class MunitoAIService:
    """Async client for Munito AI OpenAI-compatible chat completions with persistent config."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://api.munito.ai/inference/v1",
        system_prompt: str | None = None,
        timeout_seconds: float = 60.0,
        db_path: str | Path | None = None,
    ) -> None:
        self.api_key = api_key
        self.default_model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout_seconds = timeout_seconds

        self.db_path = str(db_path) if db_path is not None else resolve_default_ai_db()
        self._is_memory = self.db_path == ":memory:"
        self._lock = threading.Lock()
        self._conn = self._create_connection()
        self._init_db()

        self._config_cache: dict[int, AIChatConfig] = {}
        self._stats_cache: dict[int, AIChatStats] = {}

    @property
    def model(self) -> str:
        return self.default_model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _create_connection(self) -> sqlite3.Connection:
        if not self._is_memory and not self.db_path.startswith("file:"):
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.startswith("file:"):
            return sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock:
            if not self._is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
                self._conn.execute("PRAGMA temp_store=MEMORY;")

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_chat_settings (
                    chat_id INTEGER PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    model TEXT NOT NULL DEFAULT 'deepseek-ai/deepseek-v4.1-flash-1017',
                    system_prompt TEXT
                )
                """
            )

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_chat_usage (
                    chat_id INTEGER PRIMARY KEY,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.commit()

    def _sync_get_chat_config(self, chat_id: int) -> AIChatConfig:
        with self._lock:
            if chat_id in self._config_cache:
                return self._config_cache[chat_id]

            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT enabled, model, system_prompt FROM ai_chat_settings WHERE chat_id = ?",
                (chat_id,),
            )
            row = cursor.fetchone()
            if row:
                cfg = AIChatConfig(
                    chat_id=chat_id,
                    enabled=bool(row[0]),
                    model=str(row[1]) if row[1] else self.default_model,
                    system_prompt=row[2],
                )
            else:
                cfg = AIChatConfig(chat_id=chat_id, model=self.default_model)
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO ai_chat_settings (chat_id, enabled, model, system_prompt)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chat_id, int(cfg.enabled), cfg.model, cfg.system_prompt),
                )
                self._conn.commit()

            self._config_cache[chat_id] = cfg
            return cfg

    def _sync_update_chat_config(self, chat_id: int, **kwargs: Any) -> AIChatConfig:
        cfg = self._sync_get_chat_config(chat_id)
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

            self._conn.execute(
                """
                INSERT OR REPLACE INTO ai_chat_settings (chat_id, enabled, model, system_prompt)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, int(cfg.enabled), cfg.model, cfg.system_prompt),
            )
            self._conn.commit()
            self._config_cache[chat_id] = cfg
            return cfg

    def _sync_get_chat_stats(self, chat_id: int) -> AIChatStats:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT request_count, prompt_tokens, completion_tokens, total_tokens
                FROM ai_chat_usage WHERE chat_id = ?
                """,
                (chat_id,),
            )
            row = cursor.fetchone()
            if row:
                stats = AIChatStats(
                    chat_id=chat_id,
                    request_count=int(row[0]),
                    prompt_tokens=int(row[1]),
                    completion_tokens=int(row[2]),
                    total_tokens=int(row[3]),
                )
            else:
                stats = AIChatStats(chat_id=chat_id)
            self._stats_cache[chat_id] = stats
            return stats

    def _sync_record_usage(
        self, chat_id: int, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> AIChatStats:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO ai_chat_usage (chat_id, request_count, prompt_tokens, completion_tokens, total_tokens)
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    request_count = request_count + 1,
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens,
                    total_tokens = total_tokens + excluded.total_tokens
                """,
                (chat_id, prompt_tokens, completion_tokens, total_tokens),
            )
            self._conn.commit()
            self._stats_cache.pop(chat_id, None)

        return self._sync_get_chat_stats(chat_id)

    def _sync_reset_chat_stats(self, chat_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM ai_chat_usage WHERE chat_id = ?", (chat_id,))
            self._conn.commit()
            self._stats_cache.pop(chat_id, None)

    async def get_chat_config(self, chat_id: int) -> AIChatConfig:
        return await asyncio.to_thread(self._sync_get_chat_config, chat_id)

    async def update_chat_config(self, chat_id: int, **kwargs: Any) -> AIChatConfig:
        return await asyncio.to_thread(self._sync_update_chat_config, chat_id, **kwargs)

    async def get_chat_stats(self, chat_id: int) -> AIChatStats:
        return await asyncio.to_thread(self._sync_get_chat_stats, chat_id)

    async def record_usage(
        self, chat_id: int, prompt_tokens: int, completion_tokens: int, total_tokens: int
    ) -> AIChatStats:
        return await asyncio.to_thread(
            self._sync_record_usage, chat_id, prompt_tokens, completion_tokens, total_tokens
        )

    async def reset_chat_stats(self, chat_id: int) -> None:
        await asyncio.to_thread(self._sync_reset_chat_stats, chat_id)

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    async def generate_reply(
        self,
        prompt: str,
        context_text: str | None = None,
        image_bytes: bytes | None = None,
        image_mime: str = "image/jpeg",
        model: str | None = None,
        chat_id: int | None = None,
    ) -> str:
        """
        Generate completion from Munito AI with optional referenced text or image context.
        Tracks token usage in SQLite when chat_id is provided.
        """
        if not self.is_configured:
            raise RuntimeError("Munito API key is not configured (MUNITO_API_KEY is missing)")

        target_model = model or self.default_model
        system_prompt = self.system_prompt

        if chat_id is not None:
            cfg = await self.get_chat_config(chat_id)
            if not cfg.enabled:
                raise RuntimeError("AI асистент вимкнено для цього чату адміністратором.")
            if not model:
                target_model = cfg.model
            if cfg.system_prompt:
                system_prompt = cfg.system_prompt

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        user_content: list[dict[str, Any]] = []

        # If user referenced previous text from another message
        full_text_prompt = ""
        if context_text and context_text.strip():
            full_text_prompt += (
                f"[Повідомлення з чату, на яке посилається користувач]:\n{context_text.strip()}\n\n"
            )

        if prompt and prompt.strip():
            full_text_prompt += prompt.strip()
        elif not image_bytes:
            full_text_prompt += "Поясни або дай відповідь на це повідомлення."

        if full_text_prompt:
            user_content.append({"type": "text", "text": full_text_prompt})

        # If user attached or referenced an image
        if image_bytes:
            b64_image = base64.b64encode(image_bytes).decode("ascii")
            data_url = f"data:{image_mime};base64,{b64_image}"
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )

        if not user_content:
            return "Не вказано запитання чи повідомлення для відповіді."

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": target_model,
            "messages": messages,
            "max_tokens": 2048,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 401:
                        logger.error("Munito AI error: 401 Unauthorized (invalid API key)")
                        raise RuntimeError("Невірний або недійсний ключ Munito AI (HTTP 401).")
                    if resp.status == 429:
                        logger.warning("Munito AI error: 429 Rate limit exceeded")
                        raise RuntimeError(
                            "Перевищено ліміт запитів до Munito AI. Зачекайте хвилинку."
                        )
                    if resp.status == 503:
                        logger.warning("Munito AI error: 503 Service unavailable / no capacity")
                        raise RuntimeError(
                            "Модель зараз перевантажена на сервері Munito. Спробуйте пізніше."
                        )
                    if resp.status >= 400:
                        err_body = await resp.text()
                        logger.error("Munito AI error HTTP %d: %s", resp.status, err_body)
                        raise RuntimeError(f"Помилка сервісу Munito AI (HTTP {resp.status}).")

                    data = await resp.json()
            except aiohttp.ClientError as err:
                logger.error("Munito AI network connection error: %s", err)
                raise RuntimeError(f"Не вдалося з'єднатися з Munito AI: {err}") from err

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Munito AI повернув порожню відповідь.")

        # Record token metrics
        if chat_id is not None:
            usage = data.get("usage") or {}
            p_tok = int(usage.get("prompt_tokens") or 0)
            c_tok = int(usage.get("completion_tokens") or 0)
            t_tok = int(usage.get("total_tokens") or (p_tok + c_tok))
            await self.record_usage(chat_id, p_tok, c_tok, t_tok)

        first_choice = choices[0]
        message_obj = first_choice.get("message") or {}
        content = message_obj.get("content")

        if not content and message_obj.get("reasoning"):
            content = message_obj["reasoning"]

        if not content or not str(content).strip():
            return "<i>(Модель повернула порожню відповідь)</i>"

        return str(content).strip()
