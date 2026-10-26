from __future__ import annotations

import logging
import uuid

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
)

from filters.dice import DiceFilter
from services.dice import DiceParser, get_secret_roll, store_secret_roll
from utils.proxy import bot_proxy as bot
from utils.responses import safe_reply

logger = logging.getLogger(__name__)

dice_router = Router(name="dice")


@dice_router.message(Command("roll", "ролл", "кинь", "кидок"))
async def roll_command_handler(message: Message, command: CommandObject) -> None:
    expr = command.args.strip() if command.args else "d20"
    user_name = message.from_user.first_name if message.from_user else "Player"
    try:
        res = DiceParser.parse_and_roll(expr)
        await safe_reply(message, res.format_html(user_name), parse_mode="HTML")
    except ValueError as err:
        await safe_reply(message, f"⚠️ <b>Invalid roll syntax:</b> {err}", parse_mode="HTML")


@dice_router.message(Command("dmroll", "дмролл", "дм", "секретролл"))
async def dmroll_command_handler(message: Message, command: CommandObject) -> None:
    expr = command.args.strip() if command.args else "d20"
    user_name = message.from_user.first_name if message.from_user else "Player"
    user_id = message.from_user.id if message.from_user else 0

    try:
        res = DiceParser.parse_and_roll(expr)
    except ValueError as err:
        await safe_reply(message, f"⚠️ <b>Invalid roll syntax:</b> {err}", parse_mode="HTML")
        return

    if message.chat.type == "private":
        await safe_reply(message, res.format_html(user_name), parse_mode="HTML")
        return

    secret_id = store_secret_roll(
        user_id=user_id,
        user_name=user_name,
        result_html=res.format_html(user_name),
        result_plain=res.format_plain_text(user_name),
    )
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👁 View Secret Roll",
                    callback_data=f"secret_roll:{secret_id}",
                )
            ]
        ]
    )
    await safe_reply(
        message,
        f"🎲 <b>Secret Roll made by {user_name}</b>\n<i>Only {user_name} can view this result.</i>",
        reply_markup=keyboard,
        parse_mode="HTML",
    )


@dice_router.message(DiceFilter())
async def natural_dice_handler(message: Message, dice_expr: str, is_secret: bool = False) -> None:
    user_name = message.from_user.first_name if message.from_user else "Player"
    user_id = message.from_user.id if message.from_user else 0

    try:
        res = DiceParser.parse_and_roll(dice_expr)
    except ValueError:
        return

    if is_secret:
        if message.chat.type == "private":
            await safe_reply(message, res.format_html(user_name), parse_mode="HTML")
            return

        secret_id = store_secret_roll(
            user_id=user_id,
            user_name=user_name,
            result_html=res.format_html(user_name),
            result_plain=res.format_plain_text(user_name),
        )
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="👁 View Secret Roll",
                        callback_data=f"secret_roll:{secret_id}",
                    )
                ]
            ]
        )
        await safe_reply(
            message,
            f"🎲 <b>Secret Roll made by {user_name}</b>\n<i>Only {user_name} can view this result.</i>",
            reply_markup=keyboard,
            parse_mode="HTML",
        )
    else:
        await safe_reply(message, res.format_html(user_name), parse_mode="HTML")


@dice_router.callback_query(F.data.startswith("secret_roll:"))
async def secret_roll_callback_handler(callback_query: CallbackQuery) -> None:
    if not callback_query.data:
        return
    parts = callback_query.data.split(":")
    if len(parts) < 2:
        return
    secret_id = parts[1]

    entry = get_secret_roll(secret_id)
    if not entry:
        await callback_query.answer(
            "⚠️ This secret roll has expired or was not found.", show_alert=True
        )
        return

    roller_id, roller_name, _, result_plain = entry
    if callback_query.from_user and callback_query.from_user.id == roller_id:
        await callback_query.answer(result_plain, show_alert=True)
    else:
        await callback_query.answer(
            f"🔒 Only {roller_name} can view this secret roll!", show_alert=True
        )


@dice_router.inline_query()
async def inline_dice_handler(inline_query: InlineQuery) -> None:
    query = inline_query.query.strip()
    user_name = inline_query.from_user.first_name if inline_query.from_user else "Player"
    user_id = inline_query.from_user.id if inline_query.from_user else "unknown"
    username = inline_query.from_user.username if inline_query.from_user else "unknown"

    logger.info(
        "Received inline dice query from user %s (@%s): '%s'",
        user_id,
        username,
        query,
    )

    results: list[InlineQueryResultArticle] = []

    if not query:
        presets = [
            ("d20", "Standard d20", "Roll 1d20"),
            ("2d20kh1", "d20 with Advantage", "Roll 2d20 (Keep Highest)"),
            ("2d20kl1", "d20 with Disadvantage", "Roll 2d20 (Keep Lowest)"),
            ("d100", "Percentile d100", "Roll 1d100"),
            ("d12", "d12 Die", "Roll 1d12"),
            ("d10", "d10 Die", "Roll 1d10"),
            ("d8", "d8 Die", "Roll 1d8"),
            ("d6", "d6 Die", "Roll 1d6"),
            ("d4", "d4 Die", "Roll 1d4"),
        ]
        for expr, title, desc in presets:
            res = DiceParser.parse_and_roll(expr)
            text = res.format_html(user_name)
            results.append(
                InlineQueryResultArticle(
                    id=f"preset_{expr}",
                    title=f"🎲 {title}",
                    description=f"{desc} (Result: {res.total})",
                    input_message_content=InputTextMessageContent(
                        message_text=text,
                        parse_mode="HTML",
                    ),
                )
            )
    elif DiceParser.is_dice_query(query):
        cleaned_query, _ = DiceParser.normalize_input(query)
        try:
            res = DiceParser.parse_and_roll(cleaned_query)
            text = res.format_html(user_name)
            results.append(
                InlineQueryResultArticle(
                    id=f"roll_{uuid.uuid4().hex[:8]}",
                    title=f"🎲 Roll {res.normalized_expression}: {res.total}",
                    description=f"Details: {res.format_plain_text(user_name).replace('\n', ' • ')}",
                    input_message_content=InputTextMessageContent(
                        message_text=text,
                        parse_mode="HTML",
                    ),
                )
            )
        except ValueError as err:
            results.append(
                InlineQueryResultArticle(
                    id="err_invalid",
                    title="⚠️ Invalid Dice Expression",
                    description=str(err),
                    input_message_content=InputTextMessageContent(
                        message_text=f"⚠️ <b>Invalid roll:</b> <code>{query}</code>\n{err}",
                        parse_mode="HTML",
                    ),
                )
            )
    else:
        # Non-dice text (e.g. natural language prompts for AI assistant)
        bot_user = await bot._get_cached_bot_user(inline_query.bot) if inline_query.bot else None
        bot_tag = (
            f"@{bot_user.username} " if (bot_user and getattr(bot_user, "username", None)) else ""
        )
        results.append(
            InlineQueryResultArticle(
                id=f"ai_{uuid.uuid4().hex[:8]}",
                title=f"🤖 Ask AI: {query[:45]}",
                description=f"Send '{query}' to bot in chat",
                input_message_content=InputTextMessageContent(
                    message_text=f"{bot_tag}{query}",
                ),
            )
        )

    await inline_query.answer(results=results, cache_time=5, is_personal=True)
