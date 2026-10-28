"""
Munito AI Service — Grok-style LLM and Vision assistant via OpenAI-compatible API
with per-chat SQLite configuration, model selection, and token usage tracking.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import aiohttp

from services.db_base import SQLiteWALService, resolve_db_path
from services.dice import DiceParser
from services.munito_ui import (
    AI_TOOLS,
    AVAILABLE_MODELS,
    CONTEXT_MODE_OPTIONS,
    CONTEXT_SIZE_OPTIONS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    REASONING_EFFORT_LEVELS,
    AIChatConfig,
    AIChatStats,
    AIChatTurn,
    build_ai_keyboard,
    format_ai_card,
    get_next_context_mode,
    get_next_context_size,
    get_next_model,
    get_next_reasoning_effort,
    markdown_to_telegram_html,
    split_telegram_message,
)

logger = logging.getLogger(__name__)


def resolve_default_ai_db() -> str:
    """Resolve the default SQLite path for AI assistant, falling back to local dir if data/ is not writable."""
    return resolve_db_path("AI_DB", "ai.db", ".write_test_ai")


class MunitoAIService(SQLiteWALService):
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

        resolved = str(db_path) if db_path is not None else resolve_default_ai_db()
        super().__init__(resolved)
        self._init_db()

        self._config_cache: dict[int, AIChatConfig] = {}
        self._stats_cache: dict[int, AIChatStats] = {}

    @property
    def model(self) -> str:
        return self.default_model

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _on_close(self) -> None:
        self._config_cache.clear()
        self._stats_cache.clear()

    def _init_db(self) -> None:
        with self._lock:
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
            for col_ddl in (
                "ALTER TABLE ai_chat_settings ADD COLUMN reasoning_effort TEXT NOT NULL DEFAULT 'low'",
                "ALTER TABLE ai_chat_settings ADD COLUMN context_mode TEXT NOT NULL DEFAULT 'mentions'",
                "ALTER TABLE ai_chat_settings ADD COLUMN context_size INTEGER NOT NULL DEFAULT 10",
            ):
                with contextlib.suppress(sqlite3.OperationalError):
                    self._conn.execute(col_ddl)

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
                "SELECT enabled, model, system_prompt, reasoning_effort, context_mode, context_size "
                "FROM ai_chat_settings WHERE chat_id = ?",
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
                    INSERT OR IGNORE INTO ai_chat_settings (
                        chat_id, enabled, model, system_prompt, reasoning_effort, context_mode, context_size
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
        base_cfg = self._sync_get_chat_config(chat_id)
        with self._lock:
            valid_updates = {k: v for k, v in kwargs.items() if hasattr(base_cfg, k)}
            cfg = replace(base_cfg, **valid_updates)

            self._conn.execute(
                """
                INSERT OR REPLACE INTO ai_chat_settings (
                    chat_id, enabled, model, system_prompt, reasoning_effort, context_mode, context_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
            return [
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

    def _sync_get_history_count(self, chat_id: int) -> int:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM ai_chat_history WHERE chat_id = ?",
                (chat_id,),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def _sync_clear_chat_history(self, chat_id: int) -> int:
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

    async def _send_completion_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        chat_id: int | None,
        target_model: str,
        reasoning_info: str,
        vision_info: str,
    ) -> tuple[dict[str, Any], tuple[int, int, int]]:
        """Execute a single HTTP chat completion request against Munito API."""
        start_time = asyncio.get_event_loop().time()
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
                return data, (p_tok, c_tok, t_tok)
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

    @staticmethod
    def _execute_tool_calls(
        tool_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> None:
        """Evaluate requested tool calls (e.g. `roll_dice`) and append tool outputs to `messages`."""
        for tc in tool_calls:
            call_id = tc.get("id") or "call_default"
            func_info = tc.get("function") or {}
            func_name = func_info.get("name")
            raw_args = func_info.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}

            if func_name == "roll_dice":
                expr = str(args.get("expression") or "1d20").strip()
                reason = args.get("reason")
                try:
                    roll_res = DiceParser.parse_and_roll(expr)
                    crit_label = None
                    if roll_res.is_crit_hit:
                        crit_label = "CRITICAL HIT (Natural 20!)"
                    elif roll_res.is_crit_fail:
                        crit_label = "CRITICAL FAIL (Natural 1!)"

                    tool_output: dict[str, Any] = {
                        "expression": roll_res.normalized_expression,
                        "total": roll_res.total,
                        "constant_modifier": roll_res.constant_modifier,
                        "critical": crit_label,
                        "plain_summary": roll_res.format_plain_text(),
                        "reason": reason,
                    }
                except Exception as exc:
                    tool_output = {
                        "error": f"Failed to parse or roll dice expression '{expr}': {exc}"
                    }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": json.dumps(tool_output, ensure_ascii=False),
                    }
                )

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

        if chat_id is not None and cfg is not None and cfg.context_size > 0:
            history = await self.get_chat_history(chat_id, limit=cfg.context_size)
            for turn in history:
                if turn.role == "assistant":
                    messages.append({"role": "assistant", "content": turn.content})
                else:
                    prefix = f"[{turn.user_name}]: " if turn.user_name else ""
                    messages.append({"role": "user", "content": f"{prefix}{turn.content}"})

        full_text_prompt = ""
        if context_text and context_text.strip():
            full_text_prompt += f"[Referenced chat message]:\n{context_text.strip()}\n\n"

        if prompt and prompt.strip():
            full_text_prompt += prompt.strip()
        elif not image_bytes:
            full_text_prompt += "Explain or reply to this message."

        user_display_prompt = (
            f"[{user_name}]: {full_text_prompt}" if user_name else full_text_prompt
        )

        user_content: list[dict[str, Any]] = []
        if user_display_prompt:
            user_content.append({"type": "text", "text": user_display_prompt})

        if image_bytes:
            b64_image = base64.b64encode(image_bytes).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime};base64,{b64_image}"},
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
            "tools": AI_TOOLS,
            "tool_choice": "auto",
        }

        if "deepseek" in target_model.lower():
            raw_effort = (
                (cfg.reasoning_effort or "low").lower() if (chat_id is not None and cfg) else "low"
            )
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
            data, (p_tok, c_tok, t_tok) = await self._send_completion_request(
                session=session,
                url=url,
                payload=payload,
                headers=headers,
                chat_id=chat_id,
                target_model=target_model,
                reasoning_info=reasoning_info,
                vision_info=vision_info,
            )

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("Munito AI returned an empty response.")

            message_obj = choices[0].get("message") or {}
            tool_calls = message_obj.get("tool_calls") or []

            if tool_calls:
                logger.info(
                    "Munito AI invoked %d tool call(s) (chat_id=%s)",
                    len(tool_calls),
                    chat_id,
                )
                messages.append(message_obj)
                self._execute_tool_calls(tool_calls, messages)

                followup_payload = dict(payload)
                followup_payload["messages"] = messages
                followup_payload.pop("tools", None)
                followup_payload.pop("tool_choice", None)

                data2, (p2, c2, t2) = await self._send_completion_request(
                    session=session,
                    url=url,
                    payload=followup_payload,
                    headers=headers,
                    chat_id=chat_id,
                    target_model=target_model,
                    reasoning_info=reasoning_info,
                    vision_info="no",
                )
                p_tok += p2
                c_tok += c2
                t_tok += t2

                choices2 = data2.get("choices") or []
                if choices2:
                    message_obj = choices2[0].get("message") or {}

        if chat_id is not None:
            await self.record_usage(chat_id, p_tok, c_tok, t_tok)

        content = message_obj.get("content") or message_obj.get("reasoning")
        if not content or not str(content).strip():
            reply_result = "<i>(Model returned an empty response)</i>"
        else:
            reply_result = str(content).strip()

        if chat_id is not None and cfg is not None and cfg.context_size > 0:
            hist_user_text = (
                f"[Photo attached] {full_text_prompt}".strip() if image_bytes else full_text_prompt
            )
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


__all__ = [
    "AI_TOOLS",
    "AVAILABLE_MODELS",
    "CONTEXT_MODE_OPTIONS",
    "CONTEXT_SIZE_OPTIONS",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "REASONING_EFFORT_LEVELS",
    "AIChatConfig",
    "AIChatStats",
    "AIChatTurn",
    "MunitoAIService",
    "build_ai_keyboard",
    "format_ai_card",
    "get_next_context_mode",
    "get_next_context_size",
    "get_next_model",
    "get_next_reasoning_effort",
    "markdown_to_telegram_html",
    "resolve_default_ai_db",
    "split_telegram_message",
]
