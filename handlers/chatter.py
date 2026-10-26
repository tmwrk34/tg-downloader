from __future__ import annotations

import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from filters.admin import is_chat_admin
from services.chatter import ChatterConfig, ChatterStats
from utils.proxy import bot_proxy as bot
from utils.responses import safe_reply

logger = logging.getLogger(__name__)

chatter_router = Router(name="chatter")

REPLY_CHANCE_STEPS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
REACT_CHANCE_STEPS = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
COOLDOWN_STEPS = [10, 20, 30, 60, 120]


def _next_step(current: float | int, steps: list[Any]) -> Any:
    closest_idx = min(range(len(steps)), key=lambda i: abs(steps[i] - current))
    return steps[(closest_idx + 1) % len(steps)]


def format_chatter_card(
    chat_title: str,
    chat_id: int,
    cfg: ChatterConfig,
    stats: ChatterStats,
) -> str:
    status_str = "🟢 Enabled" if cfg.enabled else "🔴 Disabled"
    learn_str = "🟢 Enabled" if cfg.learning_enabled else "🔴 Disabled"
    mention_str = "🟢 Always (100%)" if cfg.reply_to_mention else "🔴 Disabled"

    return (
        f"🎭 <b>Tavern Chatter Settings</b>\n"
        f"<b>Chat:</b> {chat_title} (<code>{chat_id}</code>)\n\n"
        f"• <b>Status:</b> {status_str}\n"
        f"• <b>Random Reply Chance:</b> {cfg.reply_chance * 100:.0f}%\n"
        f"• <b>Reaction Chance:</b> {cfg.reaction_chance * 100:.0f}%\n"
        f"• <b>Reply on Mention/Reply:</b> {mention_str}\n"
        f"• <b>Cooldown:</b> {cfg.cooldown_seconds}s\n"
        f"• <b>Learning:</b> {learn_str}\n"
        f"• <b>Vocabulary:</b> {stats.total_transitions:,} transitions ({stats.unique_words:,} unique words)\n\n"
        f"<i>Tap the buttons below to tweak settings or use commands:</i>\n"
        f"<code>/chatter on|off</code>\n"
        f"<code>/chatter chance 0-100</code>\n"
        f"<code>/chatter react 0-100</code>\n"
        f"<code>/chatter cooldown &lt;sec&gt;</code>\n"
        f"<code>/chatter clear</code>"
    )


def build_chatter_keyboard(
    chat_id: int,
    cfg: ChatterConfig,
    confirm_clear: bool = False,
) -> InlineKeyboardMarkup:
    if confirm_clear:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="⚠️ Confirm Clear All Words",
                        callback_data=f"chatter:clear_confirm:{chat_id}",
                    ),
                    InlineKeyboardButton(
                        text="❌ Cancel",
                        callback_data=f"chatter:refresh:{chat_id}",
                    ),
                ]
            ]
        )

    toggle_text = "🔴 Turn OFF" if cfg.enabled else "🟢 Turn ON"
    reply_pct = int(cfg.reply_chance * 100)
    react_pct = int(cfg.reaction_chance * 100)
    learn_text = "🧠 Learn: ON" if cfg.learning_enabled else "🧠 Learn: OFF"

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Status: {toggle_text}",
                    callback_data=f"chatter:toggle:{chat_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"🎲 Reply: {reply_pct}%",
                    callback_data=f"chatter:chance:{chat_id}",
                ),
                InlineKeyboardButton(
                    text=f"🤡 React: {react_pct}%",
                    callback_data=f"chatter:react:{chat_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=f"⏳ Cooldown: {cfg.cooldown_seconds}s",
                    callback_data=f"chatter:cooldown:{chat_id}",
                ),
                InlineKeyboardButton(
                    text=learn_text,
                    callback_data=f"chatter:learn:{chat_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="🧹 Clear Words",
                    callback_data=f"chatter:clear_prompt:{chat_id}",
                ),
                InlineKeyboardButton(
                    text="🔄 Refresh",
                    callback_data=f"chatter:refresh:{chat_id}",
                ),
            ],
        ]
    )


@chatter_router.message(Command("chatter", "бард", "дурачок", "балабол"))
async def chatter_command_handler(message: Message, command: CommandObject) -> None:
    user_id = message.from_user.id if message.from_user else 0
    chat_id = message.chat.id
    chat_title = message.chat.title or message.chat.first_name or "Private Chat"

    if not await is_chat_admin(message.bot, chat_id, user_id):
        await safe_reply(
            message, "⚠️ Only chat administrators can configure tavern chatter settings."
        )
        return

    args = (command.args or "").strip().split()
    cfg = await bot.chatter_service.get_config(chat_id)

    if args:
        subcmd = args[0].lower()
        if subcmd in {"on", "start", "enable", "1"}:
            cfg = await bot.chatter_service.update_config(chat_id, enabled=True)
        elif subcmd in {"off", "stop", "disable", "0"}:
            cfg = await bot.chatter_service.update_config(chat_id, enabled=False)
        elif subcmd in {"chance", "reply"} and len(args) > 1:
            try:
                pct = float(args[1].rstrip("%"))
                if not (0.0 <= pct <= 100.0):
                    raise ValueError
                cfg = await bot.chatter_service.update_config(chat_id, reply_chance=pct / 100.0)
            except ValueError:
                await safe_reply(
                    message,
                    "⚠️ Please provide a valid percentage between 0 and 100 (e.g. <code>/chatter chance 15</code>).",
                    parse_mode="HTML",
                )
                return
        elif subcmd in {"react", "reaction"} and len(args) > 1:
            try:
                pct = float(args[1].rstrip("%"))
                if not (0.0 <= pct <= 100.0):
                    raise ValueError
                cfg = await bot.chatter_service.update_config(chat_id, reaction_chance=pct / 100.0)
            except ValueError:
                await safe_reply(
                    message,
                    "⚠️ Please provide a valid percentage between 0 and 100 (e.g. <code>/chatter react 10</code>).",
                    parse_mode="HTML",
                )
                return
        elif subcmd in {"cooldown", "delay", "cd"} and len(args) > 1:
            try:
                cd = int(args[1])
                if cd < 0:
                    raise ValueError
                cfg = await bot.chatter_service.update_config(chat_id, cooldown_seconds=cd)
            except ValueError:
                await safe_reply(
                    message,
                    "⚠️ Please provide a valid cooldown in seconds (e.g. <code>/chatter cooldown 30</code>).",
                    parse_mode="HTML",
                )
                return
        elif subcmd in {"learn", "learning"}:
            val = None
            if len(args) > 1:
                val = args[1].lower() in {"on", "enable", "true", "1"}
            else:
                val = not cfg.learning_enabled
            cfg = await bot.chatter_service.update_config(chat_id, learning_enabled=val)
        elif subcmd in {"clear", "reset"}:
            cleared = await bot.chatter_service.clear_vocabulary(chat_id)
            await safe_reply(message, f"🧹 Cleared {cleared:,} words from this chat's vocabulary!")
            return

    stats = await bot.chatter_service.get_stats(chat_id)
    text = format_chatter_card(chat_title, chat_id, cfg, stats)
    keyboard = build_chatter_keyboard(chat_id, cfg)
    await safe_reply(message, text, reply_markup=keyboard, parse_mode="HTML")


@chatter_router.callback_query(F.data.startswith("chatter:"))
async def chatter_callback_handler(callback: CallbackQuery) -> None:
    if not callback.data:
        return

    parts = callback.data.split(":")
    if len(parts) < 3:
        return

    _, action, chat_id_str = parts[:3]
    try:
        target_chat_id = int(chat_id_str)
    except ValueError:
        return

    user_id = callback.from_user.id if callback.from_user else 0
    if not await is_chat_admin(callback.bot, target_chat_id, user_id):
        await callback.answer(
            "⚠️ Only chat administrators can modify chatter settings.",
            show_alert=True,
        )
        return

    cfg = await bot.chatter_service.get_config(target_chat_id)
    confirm_clear = False

    if action == "toggle":
        cfg = await bot.chatter_service.update_config(target_chat_id, enabled=not cfg.enabled)
        await callback.answer(f"Status changed to: {'Enabled' if cfg.enabled else 'Disabled'}")
    elif action == "chance":
        new_chance = _next_step(cfg.reply_chance, REPLY_CHANCE_STEPS)
        cfg = await bot.chatter_service.update_config(target_chat_id, reply_chance=new_chance)
        await callback.answer(f"Reply chance set to {int(new_chance * 100)}%")
    elif action == "react":
        new_react = _next_step(cfg.reaction_chance, REACT_CHANCE_STEPS)
        cfg = await bot.chatter_service.update_config(target_chat_id, reaction_chance=new_react)
        await callback.answer(f"Reaction chance set to {int(new_react * 100)}%")
    elif action == "cooldown":
        new_cd = _next_step(cfg.cooldown_seconds, COOLDOWN_STEPS)
        cfg = await bot.chatter_service.update_config(target_chat_id, cooldown_seconds=new_cd)
        await callback.answer(f"Cooldown set to {new_cd}s")
    elif action == "learn":
        cfg = await bot.chatter_service.update_config(
            target_chat_id, learning_enabled=not cfg.learning_enabled
        )
        await callback.answer(f"Learning {'Enabled' if cfg.learning_enabled else 'Disabled'}")
    elif action == "clear_prompt":
        confirm_clear = True
        await callback.answer()
    elif action == "clear_confirm":
        cleared = await bot.chatter_service.clear_vocabulary(target_chat_id)
        await callback.answer(f"Cleared {cleared:,} words!", show_alert=True)
    elif action == "refresh":
        await callback.answer("Refreshed!")

    target_msg = callback.message if isinstance(callback.message, Message) else None
    if target_msg:
        chat_title = target_msg.chat.title or target_msg.chat.first_name or "Chat"
        stats = await bot.chatter_service.get_stats(target_chat_id)
        text = format_chatter_card(chat_title, target_chat_id, cfg, stats)
        keyboard = build_chatter_keyboard(target_chat_id, cfg, confirm_clear=confirm_clear)
        try:
            await target_msg.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
        except Exception:
            pass
