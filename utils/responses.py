from __future__ import annotations

import logging
from typing import Any

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

logger = logging.getLogger(__name__)


async def _safe_send(
    message: Message,
    reply_method_name: str,
    answer_method_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Invoke a `message.reply_*` method, automatically falling back to `message.answer_*`
    if the original message was deleted in the chat ('message to be replied not found').
    """
    reply_fn = getattr(message, reply_method_name)
    try:
        return await reply_fn(*args, **kwargs)
    except TelegramBadRequest as e:
        if "message to be replied not found" in str(e).lower():
            answer_fn = getattr(message, answer_method_name)
            try:
                return await answer_fn(*args, **kwargs)
            except Exception as answer_err:
                logger.debug("%s fallback failed: %s", reply_method_name, answer_err)
                return None
        raise


async def safe_reply(message: Message, text: str, **kwargs: Any) -> Message | None:
    """
    Reply to a message, automatically falling back to message.answer()
    if the original message was deleted in the chat (message to be replied not found).
    """
    return await _safe_send(message, "reply", "answer", text, **kwargs)


async def safe_reply_video(message: Message, **kwargs: Any) -> Message | None:
    """Send video as reply, falling back to answer_video if original message was deleted."""
    return await _safe_send(message, "reply_video", "answer_video", **kwargs)


async def safe_reply_photo(message: Message, **kwargs: Any) -> Message | None:
    """Send photo as reply, falling back to answer_photo if original message was deleted."""
    return await _safe_send(message, "reply_photo", "answer_photo", **kwargs)


async def safe_reply_audio(message: Message, **kwargs: Any) -> Message | None:
    """Send audio as reply, falling back to answer_audio if original message was deleted."""
    return await _safe_send(message, "reply_audio", "answer_audio", **kwargs)


async def safe_reply_media_group(message: Message, **kwargs: Any) -> list[Message] | None:
    """Send media group as reply, falling back to answer_media_group if original message was deleted."""
    return await _safe_send(message, "reply_media_group", "answer_media_group", **kwargs)
