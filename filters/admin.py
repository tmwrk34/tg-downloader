from __future__ import annotations

import logging
import sys

from aiogram import Bot
from aiogram.filters import Filter
from aiogram.types import CallbackQuery, Message

logger = logging.getLogger(__name__)


async def is_chat_admin(active_bot: Bot | None, chat_id: int, user_id: int) -> bool:
    """Verify if user is a chat administrator, group creator, or bot superadmin."""
    bot_mod = sys.modules.get("bot")
    cfg = getattr(bot_mod, "config", None)
    admin_ids = getattr(cfg, "admin_user_ids", set()) if cfg else set()

    if user_id in admin_ids:
        return True
    if chat_id > 0 and user_id == chat_id:
        return True
    if active_bot is None:
        return True
    try:
        member = await active_bot.get_chat_member(chat_id=chat_id, user_id=user_id)
        return member.status in {"creator", "administrator"}
    except Exception as e:
        logger.debug("Could not verify chat admin status for %d in %d: %s", user_id, chat_id, e)
        return False


class IsChatAdminFilter(Filter):
    """aiogram filter ensuring the interacting user is an admin in the context chat."""

    async def __call__(self, event: Message | CallbackQuery, bot: Bot) -> bool:
        user = event.from_user
        if not user:
            return False
        chat = (
            event.chat
            if isinstance(event, Message)
            else (event.message.chat if event.message else None)
        )
        chat_id = chat.id if chat else 0
        return await is_chat_admin(bot, chat_id, user.id)
