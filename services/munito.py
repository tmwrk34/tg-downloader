"""
Munito AI Service — Grok-style LLM and Vision assistant via OpenAI-compatible API
with per-chat SQLite configuration, model selection, and token usage tracking.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import html
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, witty, and concise AI assistant in a Telegram group chat. "
    "Format your answers clearly, be brief and direct, and avoid unnecessary filler or repetition. "
    "Respond in the language of the user's prompt (Ukrainian by default). "
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

REASONING_EFFORT_LEVELS = ["low", "medium", "high"]
CONTEXT_SIZE_OPTIONS = [0, 5, 10, 15, 20]
CONTEXT_MODE_OPTIONS = ["mentions", "all"]


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
    reasoning_effort: str = "low"
    context_mode: str = "mentions"  # "mentions" | "all"
    context_size: int = 10  # number of recent messages to inject (0 = disabled)


@dataclass(frozen=True)
class AIChatTurn:
    id: int
    chat_id: int
    user_id: int | None
    user_name: str | None
    role: str
    content: str
    has_image: bool
    created_at: float


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


def get_next_reasoning_effort(current_effort: str) -> str:
    """Cycle reasoning effort levels: low -> none -> high -> low."""
    try:
        idx = REASONING_EFFORT_LEVELS.index(current_effort.lower())
        return REASONING_EFFORT_LEVELS[(idx + 1) % len(REASONING_EFFORT_LEVELS)]
    except ValueError:
        return REASONING_EFFORT_LEVELS[0]


def get_next_context_size(current_size: int) -> int:
    """Cycle context size: 0 -> 5 -> 10 -> 15 -> 20 -> 0."""
    try:
        idx = CONTEXT_SIZE_OPTIONS.index(current_size)
        return CONTEXT_SIZE_OPTIONS[(idx + 1) % len(CONTEXT_SIZE_OPTIONS)]
    except ValueError:
        return 10


def get_next_context_mode(current_mode: str) -> str:
    """Toggle context collection mode between 'mentions' and 'all'."""
    return "all" if current_mode.lower() == "mentions" else "mentions"


def format_ai_card(
    chat_title: str,
    chat_id: int,
    cfg: AIChatConfig,
    stats: AIChatStats,
    history_count: int = 0,
) -> str:
    """Format AI assistant status, model, context memory, and token usage into an HTML message."""
    status_str = "🟢 Enabled" if cfg.enabled else "🔴 Disabled"
    effort_labels = {
        "low": "⚡ Low (Fast, ~15s)",
        "none": "⚡ Low (Fast, ~15s)",
        "medium": "⚖️ Medium (~35s)",
        "high": "🧠 High (Deep CoT, ~90s)",
    }
    effort_display = effort_labels.get(cfg.reasoning_effort, cfg.reasoning_effort)
    mode_labels = {
        "mentions": "🎯 Mentions Only",
        "all": "🌐 All Chat Messages",
    }
    mode_display = mode_labels.get(cfg.context_mode, cfg.context_mode)

    if cfg.context_size > 0:
        memory_str = f"🟢 Active ({history_count}/{cfg.context_size} msgs, {mode_display})"
    else:
        memory_str = "⚪ Disabled (0 msgs)"

    return (
        f"🤖 <b>Munito AI Assistant Settings</b>\n"
        f"<b>Chat:</b> <code>{chat_title}</code> (<code>{chat_id}</code>)\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ <b>Status:</b> {status_str}\n"
        f"🧠 <b>Model:</b> <code>{cfg.model}</code>\n"
        f"💭 <b>Thinking:</b> <code>{effort_display}</code>\n"
        f"💬 <b>Memory:</b> <code>{memory_str}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Token Usage:</b>\n"
        f"  • Total requests: <b>{stats.request_count:,}</b>\n"
        f"  • Prompt tokens: <b>{stats.prompt_tokens:,}</b>\n"
        f"  • Generated tokens: <b>{stats.completion_tokens:,}</b>\n"
        f"  • Total tokens: <b>{stats.total_tokens:,}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Triggered exclusively via bot tag: @bot &lt;prompt&gt;</i>"
    )


def build_ai_keyboard(
    chat_id: int, cfg: AIChatConfig, confirm_reset: bool = False
) -> InlineKeyboardMarkup:
    """Construct inline keyboard for AI configuration."""
    toggle_text = "🔴 Disable" if cfg.enabled else "🟢 Enable"
    model_short = cfg.model.split("/")[-1]
    mode_short = "🎯 Mentions" if cfg.context_mode == "mentions" else "🌐 All Chat"
    ctx_text = f"💬 Context: {cfg.context_size} msgs" if cfg.context_size > 0 else "💬 Context: Off"

    rows = [
        [
            InlineKeyboardButton(
                text=f"Status: {toggle_text}",
                callback_data=f"ai:toggle:{chat_id}",
            )
        ],
        [
            InlineKeyboardButton(
                text=f"🧠 Model: {model_short}",
                callback_data=f"ai:model:{chat_id}",
            )
        ],
        [
            InlineKeyboardButton(
                text=f"💭 Thinking: {cfg.reasoning_effort.title()}",
                callback_data=f"ai:effort:{chat_id}",
            )
        ],
        [
            InlineKeyboardButton(
                text=ctx_text,
                callback_data=f"ai:ctx_size:{chat_id}",
            ),
            InlineKeyboardButton(
                text=f"Mode: {mode_short}",
                callback_data=f"ai:ctx_mode:{chat_id}",
            ),
        ],
        [
            InlineKeyboardButton(
                text="🧠 Forget Context",
                callback_data=f"ai:clear_mem:{chat_id}",
            )
        ],
    ]

    if confirm_reset:
        rows.append(
            [
                InlineKeyboardButton(
                    text="⚠️ Confirm Reset Stats",
                    callback_data=f"ai:reset_confirm:{chat_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Cancel",
                    callback_data=f"ai:refresh:{chat_id}",
                ),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    text="🧹 Reset Token Stats",
                    callback_data=f"ai:reset_prompt:{chat_id}",
                )
            ]
        )

    rows.append(
        [
            InlineKeyboardButton(
                text="🔄 Refresh",
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


def markdown_to_telegram_html(text: str) -> str:
    """
    Convert Markdown text from LLM completions into valid Telegram HTML format.
    Safely escapes raw HTML characters outside of tags, formats code blocks,
    inline code, bold, italic, strikethrough, links, and bullet points.
    """
    if not text:
        return ""

    # Placeholders for code blocks and inline code to prevent accidental markup collision
    placeholders: list[str] = []

    def save_placeholder(content: str) -> str:
        idx = len(placeholders)
        placeholders.append(content)
        return f"\x00CODE{idx}\x00"

    # 1. Protect code blocks: ```lang\ncode\n```
    def replace_code_block(match: re.Match) -> str:
        lang = match.group(1) or ""
        code = match.group(2)
        code_escaped = html.escape(code, quote=False)
        if lang:
            tag = f'<pre><code class="language-{html.escape(lang.strip(), quote=True)}">{code_escaped}</code></pre>'
        else:
            tag = f"<pre><code>{code_escaped}</code></pre>"
        return save_placeholder(tag)

    result = re.sub(r"```([a-zA-Z0-9_\-\+]*)\n?([\s\S]*?)```", replace_code_block, text)

    # 2. Protect inline code: `code`
    def replace_inline_code(match: re.Match) -> str:
        code = match.group(1)
        return save_placeholder(f"<code>{html.escape(code, quote=False)}</code>")

    result = re.sub(r"`([^`\n]+)`", replace_inline_code, result)

    # 3. Escape HTML characters in remaining text (&, <, >)
    result = html.escape(result, quote=False)

    # 4. Headers: # Title -> <b>Title</b>
    result = re.sub(r"(?m)^#{1,6}\s+(.+)$", r"<b>\1</b>", result)

    # 5. Bold: **text** or __text__
    result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"__(.+?)__", r"<b>\1</b>", result)

    # 6. Italic: *text* or _text_
    result = re.sub(r"(?<!\w)\*([^\*\n]+?)\*(?!\w)", r"<i>\1</i>", result)
    result = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", result)

    # 7. Strikethrough: ~~text~~
    result = re.sub(r"~~(.+?)~~", r"<s>\1</s>", result)

    # 8. Links: [label](url)
    def replace_link(match: re.Match) -> str:
        label = match.group(1)
        url = match.group(2)
        return f'<a href="{url}">{label}</a>'

    result = re.sub(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", replace_link, result)

    # 9. List bullets: convert leading "- " or "* " to clean Unicode bullet "• "
    result = re.sub(r"(?m)^[\*\-]\s+", "• ", result)

    # 10. Restore code blocks and inline code
    for idx, content in enumerate(placeholders):
        result = result.replace(f"\x00CODE{idx}\x00", content)

    return result


class MunitoAIService:
    """Async client for Munito AI OpenAI-compatible chat completions with persistent config."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://api.munito.ai/inference/v1",
        system_prompt: str | None = None,
        timeout_seconds: float = 180.0,
        db_path: str | Path | None = None,
    ) -> None:
        self.api_key = api_key
        self.default_model = model
        clean_base = (base_url or "https://api.munito.ai/inference/v1").strip().rstrip("/")
        if clean_base.endswith("/chat/completions"):
            clean_base = clean_base.removesuffix("/chat/completions").rstrip("/")
        self.base_url = clean_base
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
                    system_prompt TEXT,
                    reasoning_effort TEXT NOT NULL DEFAULT 'low',
                    context_mode TEXT NOT NULL DEFAULT 'mentions',
                    context_size INTEGER NOT NULL DEFAULT 10
                )
                """
            )
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(
                    "ALTER TABLE ai_chat_settings ADD COLUMN reasoning_effort TEXT NOT NULL DEFAULT 'low'"
                )
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(
                    "ALTER TABLE ai_chat_settings ADD COLUMN context_mode TEXT NOT NULL DEFAULT 'mentions'"
                )
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute(
                    "ALTER TABLE ai_chat_settings ADD COLUMN context_size INTEGER NOT NULL DEFAULT 10"
                )

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER,
                    user_name TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    has_image INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_history_chat ON ai_chat_history (chat_id, id);"
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
                "SELECT enabled, model, system_prompt, reasoning_effort, context_mode, context_size FROM ai_chat_settings WHERE chat_id = ?",
                (chat_id,),
            )
            row = cursor.fetchone()
            if row:
                effort = str(row[3]) if len(row) > 3 and row[3] else "low"
                ctx_mode = str(row[4]) if len(row) > 4 and row[4] else "mentions"
                ctx_size = int(row[5]) if len(row) > 5 and row[5] is not None else 10
                cfg = AIChatConfig(
                    chat_id=chat_id,
                    enabled=bool(row[0]),
                    model=str(row[1]) if row[1] else self.default_model,
                    system_prompt=row[2],
                    reasoning_effort=effort,
                    context_mode=ctx_mode,
                    context_size=ctx_size,
                )
            else:
                cfg = AIChatConfig(
                    chat_id=chat_id,
                    model=self.default_model,
                    reasoning_effort="low",
                    context_mode="mentions",
                    context_size=10,
                )
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO ai_chat_settings (chat_id, enabled, model, system_prompt, reasoning_effort, context_mode, context_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        int(cfg.enabled),
                        cfg.model,
                        cfg.system_prompt,
                        cfg.reasoning_effort,
                        cfg.context_mode,
                        cfg.context_size,
                    ),
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
                INSERT OR REPLACE INTO ai_chat_settings (chat_id, enabled, model, system_prompt, reasoning_effort, context_mode, context_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    int(cfg.enabled),
                    cfg.model,
                    cfg.system_prompt,
                    cfg.reasoning_effort,
                    cfg.context_mode,
                    cfg.context_size,
                ),
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

    def _sync_record_turn(
        self,
        chat_id: int,
        role: str,
        content: str,
        user_id: int | None = None,
        user_name: str | None = None,
        has_image: bool = False,
    ) -> None:
        """Record a single dialogue turn into ai_chat_history with pruning to 50 entries."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ai_chat_history (chat_id, user_id, user_name, role, content, has_image, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (chat_id, user_id, user_name, role, content, int(has_image), time.time()),
            )
            # Prune older messages per chat to keep table size bounded
            self._conn.execute(
                """
                DELETE FROM ai_chat_history
                WHERE chat_id = ? AND id NOT IN (
                    SELECT id FROM ai_chat_history WHERE chat_id = ? ORDER BY id DESC LIMIT 50
                )
                """,
                (chat_id, chat_id),
            )
            self._conn.commit()

    def _sync_get_chat_history(self, chat_id: int, limit: int = 10) -> list[AIChatTurn]:
        """Fetch the most recent turns for a chat in chronological order."""
        if limit <= 0:
            return []
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, chat_id, user_id, user_name, role, content, has_image, created_at
                FROM ai_chat_history
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            rows = cursor.fetchall()
            turns = [
                AIChatTurn(
                    id=row[0],
                    chat_id=row[1],
                    user_id=row[2],
                    user_name=row[3],
                    role=row[4],
                    content=row[5],
                    has_image=bool(row[6]),
                    created_at=row[7],
                )
                for row in reversed(rows)
            ]
            return turns

    def _sync_get_history_count(self, chat_id: int) -> int:
        """Get the total number of stored history entries for a chat."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM ai_chat_history WHERE chat_id = ?",
                (chat_id,),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def _sync_clear_chat_history(self, chat_id: int) -> int:
        """Delete all history entries for a chat and return count deleted."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM ai_chat_history WHERE chat_id = ?", (chat_id,))
            deleted = cursor.rowcount
            self._conn.commit()
            return deleted

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

    async def record_turn(
        self,
        chat_id: int,
        role: str,
        content: str,
        user_id: int | None = None,
        user_name: str | None = None,
        has_image: bool = False,
    ) -> None:
        await asyncio.to_thread(
            self._sync_record_turn,
            chat_id,
            role,
            content,
            user_id,
            user_name,
            has_image,
        )

    async def get_chat_history(self, chat_id: int, limit: int = 10) -> list[AIChatTurn]:
        return await asyncio.to_thread(self._sync_get_chat_history, chat_id, limit)

    async def get_history_count(self, chat_id: int) -> int:
        return await asyncio.to_thread(self._sync_get_history_count, chat_id)

    async def clear_chat_history(self, chat_id: int) -> int:
        return await asyncio.to_thread(self._sync_clear_chat_history, chat_id)

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
        user_id: int | None = None,
        user_name: str | None = None,
    ) -> str:
        """
        Generate completion from Munito AI with optional referenced text or image context.
        Tracks token usage in SQLite and maintains conversation memory when chat_id is provided.
        """
        if not self.is_configured:
            raise RuntimeError("Munito API key is not configured (MUNITO_API_KEY is missing)")

        target_model = model or self.default_model
        system_prompt = self.system_prompt
        cfg: AIChatConfig | None = None

        if chat_id is not None:
            cfg = await self.get_chat_config(chat_id)
            if not cfg.enabled:
                raise RuntimeError("AI assistant is disabled for this chat by administrator.")
            if not model:
                target_model = cfg.model
            if cfg.system_prompt:
                system_prompt = cfg.system_prompt

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Inject conversation history if context memory is active for this chat
        if chat_id is not None and cfg is not None and cfg.context_size > 0:
            history = await self.get_chat_history(chat_id, limit=cfg.context_size)
            for turn in history:
                if turn.role == "assistant":
                    messages.append({"role": "assistant", "content": turn.content})
                else:
                    prefix = f"[{turn.user_name}]: " if turn.user_name else ""
                    messages.append({"role": "user", "content": f"{prefix}{turn.content}"})

        user_content: list[dict[str, Any]] = []

        # If user referenced previous text from another message
        full_text_prompt = ""
        if context_text and context_text.strip():
            full_text_prompt += f"[Referenced chat message]:\n{context_text.strip()}\n\n"

        if prompt and prompt.strip():
            full_text_prompt += prompt.strip()
        elif not image_bytes:
            full_text_prompt += "Explain or reply to this message."

        # Prepend sender name if available so the model distinguishes different speakers in groups
        user_display_prompt = full_text_prompt
        if user_name:
            user_display_prompt = f"[{user_name}]: {full_text_prompt}"

        if user_display_prompt:
            user_content.append({"type": "text", "text": user_display_prompt})

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
            return "No prompt or message was provided to answer."

        messages.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "max_tokens": 2048,
            "stream": False,
        }

        if "deepseek" in target_model.lower():
            raw_effort = (cfg.reasoning_effort or "low").lower() if chat_id is not None else "low"
            # OpenAI / Munito reasoning_effort strictly supports 'low', 'medium', 'high'
            effort = "low" if raw_effort in {"none", "low"} else raw_effort
            if effort not in {"low", "medium", "high"}:
                effort = "low"
            payload["reasoning_effort"] = effort
            if raw_effort in {"none", "low"} and messages and messages[0].get("role") == "system":
                messages[0]["content"] += (
                    "\nKeep responses concise and direct without unnecessary filler."
                )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(
            total=self.timeout_seconds,
            connect=30.0,
            sock_read=self.timeout_seconds,
        )
        start_time = asyncio.get_event_loop().time()
        vision_info = f"yes ({len(image_bytes) // 1024}KB)" if image_bytes else "no"
        reasoning_info = payload.get("reasoning_effort", "n/a")

        logger.info(
            "Munito AI request started (chat_id=%s, model=%s, reasoning_effort=%s, vision=%s, max_tokens=%s)",
            chat_id,
            target_model,
            reasoning_info,
            vision_info,
            payload.get("max_tokens"),
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 401:
                        logger.error("Munito AI error: 401 Unauthorized (invalid API key)")
                        raise RuntimeError("Invalid or unauthorized Munito API key (HTTP 401).")
                    if resp.status == 429:
                        logger.warning("Munito AI error: 429 Rate limit exceeded")
                        raise RuntimeError(
                            "Munito AI rate limit reached. Please wait a minute before retrying."
                        )
                    if resp.status == 503:
                        logger.warning("Munito AI error: 503 Service unavailable / no capacity")
                        raise RuntimeError(
                            "The model is currently overloaded on Munito servers. Please try again later."
                        )
                    if resp.status >= 400:
                        err_body = await resp.text()
                        logger.error("Munito AI error HTTP %d: %s", resp.status, err_body)
                        raise RuntimeError(f"Munito AI service error (HTTP {resp.status}).")

                    data = await resp.json()
                    elapsed = asyncio.get_event_loop().time() - start_time
                    usage = data.get("usage") or {}
                    p_tok = int(usage.get("prompt_tokens") or 0)
                    c_tok = int(usage.get("completion_tokens") or 0)
                    t_tok = int(usage.get("total_tokens") or (p_tok + c_tok))

                    logger.info(
                        "Munito AI responded in %.2fs (chat_id=%s, model=%s, reasoning_effort=%s, vision=%s, tokens: total=%d [prompt=%d, completion=%d])",
                        elapsed,
                        chat_id,
                        target_model,
                        reasoning_info,
                        vision_info,
                        t_tok,
                        p_tok,
                        c_tok,
                    )
            except TimeoutError as err:
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.error(
                    "Munito AI request timed out after %.2fs (chat_id=%s, model=%s, reasoning_effort=%s, vision=%s)",
                    elapsed,
                    chat_id,
                    target_model,
                    reasoning_info,
                    vision_info,
                )
                raise RuntimeError(
                    f"Munito AI request timed out after {int(elapsed)}s. "
                    "The model is taking too long to think or process the image. "
                    "Try lowering thinking effort (/ai effort low) or switching model (/ai)."
                ) from err
            except aiohttp.ClientError as err:
                logger.error("Munito AI network connection error: %s", err)
                raise RuntimeError(f"Failed to connect to Munito AI: {err}") from err

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Munito AI returned an empty response.")

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
            reply_result = "<i>(Model returned an empty response)</i>"
        else:
            reply_result = str(content).strip()

        # Record conversation turns in history if memory is active
        if chat_id is not None and cfg is not None and cfg.context_size > 0:
            hist_user_text = full_text_prompt
            if image_bytes:
                hist_user_text = f"[Photo attached] {full_text_prompt}".strip()
            await self.record_turn(
                chat_id=chat_id,
                role="user",
                content=hist_user_text,
                user_id=user_id,
                user_name=user_name,
                has_image=bool(image_bytes),
            )
            await self.record_turn(
                chat_id=chat_id,
                role="assistant",
                content=reply_result,
                user_id=None,
                user_name="Bard-Bot",
                has_image=False,
            )

        return reply_result
