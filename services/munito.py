"""
Munito AI Service — Grok-style LLM and Vision assistant via OpenAI-compatible API.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, witty, and concise AI assistant in a Telegram group chat. "
    "Format your answers clearly and respond in the language of the user's prompt (Ukrainian by default). "
    "If analyzing an image or text from chat, be accurate and informative."
)


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
    """Async client for Munito AI OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-ai/deepseek-v4.1-flash-1017",
        base_url: str = "https://api.munito.ai/inference/v1",
        system_prompt: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    async def generate_reply(
        self,
        prompt: str,
        context_text: str | None = None,
        image_bytes: bytes | None = None,
        image_mime: str = "image/jpeg",
        model: str | None = None,
    ) -> str:
        """
        Generate completion from Munito AI with optional referenced text or image context.
        """
        if not self.is_configured:
            raise RuntimeError("Munito API key is not configured (MUNITO_API_KEY is missing)")

        target_model = model or self.model
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

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

        first_choice = choices[0]
        message_obj = first_choice.get("message") or {}
        content = message_obj.get("content")

        if not content and message_obj.get("reasoning"):
            content = message_obj["reasoning"]

        if not content or not str(content).strip():
            return "<i>(Модель повернула порожню відповідь)</i>"

        return str(content).strip()
