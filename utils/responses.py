from __future__ import annotations

import logging
from typing import Any

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

logger = logging.getLogger(__name__)


async def safe_reply(message: Message, text: str, **kwargs: Any) -> Message | None:
    """
    Reply to a message, automatically falling back to message.answer()
    if the original message was deleted in the chat (message to be replied not found).
    """
    try:
        return await message.reply(text, **kwargs)
    except TelegramBadRequest as e:
        if "message to be replied not found" in str(e).lower():
            try:
                return await message.answer(text, **kwargs)
            except Exception as answer_err:
                logger.debug("safe_reply fallback answer failed: %s", answer_err)
                return None
        raise


async def safe_reply_video(message: Message, **kwargs: Any) -> Message | None:
    """Send video as reply, falling back to answer_video if original message was deleted."""
    try:
        return await message.reply_video(**kwargs)
    except TelegramBadRequest as e:
        if "message to be replied not found" in str(e).lower():
            try:
                return await message.answer_video(**kwargs)
            except Exception as answer_err:
                logger.debug("safe_reply_video fallback failed: %s", answer_err)
                return None
        raise


async def safe_reply_photo(message: Message, **kwargs: Any) -> Message | None:
    """Send photo as reply, falling back to answer_photo if original message was deleted."""
    try:
        return await message.reply_photo(**kwargs)
    except TelegramBadRequest as e:
        if "message to be replied not found" in str(e).lower():
            try:
                return await message.answer_photo(**kwargs)
            except Exception as answer_err:
                logger.debug("safe_reply_photo fallback failed: %s", answer_err)
                return None
        raise


async def safe_reply_audio(message: Message, **kwargs: Any) -> Message | None:
    """Send audio as reply, falling back to answer_audio if original message was deleted."""
    try:
        return await message.reply_audio(**kwargs)
    except TelegramBadRequest as e:
        if "message to be replied not found" in str(e).lower():
            try:
                return await message.answer_audio(**kwargs)
            except Exception as answer_err:
                logger.debug("safe_reply_audio fallback failed: %s", answer_err)
                return None
        raise


async def safe_reply_media_group(message: Message, **kwargs: Any) -> list[Message] | None:
    """Send media group as reply, falling back to answer_media_group if original message was deleted."""
    try:
        return await message.reply_media_group(**kwargs)
    except TelegramBadRequest as e:
        if "message to be replied not found" in str(e).lower():
            try:
                return await message.answer_media_group(**kwargs)
            except Exception as answer_err:
                logger.debug("safe_reply_media_group fallback failed: %s", answer_err)
                return None
        raise
