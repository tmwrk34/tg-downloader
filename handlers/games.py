"""
Telegram Bot handlers for Tavern Games Hub & Texas Hold'em.
Provides /games hub, /poker table card with WebApp launch button,
and private card peek alerts (show_alert=True) directly in chat!
"""

from __future__ import annotations

import html
import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    WebAppInfo,
)

from services.games.manager import games_manager
from services.games.poker.evaluator import evaluate_hand
from services.games.poker.models import MoveType
from utils.proxy import bot_proxy as bot
from utils.responses import safe_reply

logger = logging.getLogger(__name__)

games_router = Router(name="games")


def _get_webapp_public_url() -> str | None:
    cfg = getattr(bot, "config", None)
    return getattr(cfg, "webapp_public_url", None)


def _build_poker_card_html(table_view: dict) -> str:
    pot = table_view["pot"]
    stage = table_view["stage"]
    last_action = table_view["last_action"]
    cc = (
        " ".join(table_view["community_cards"])
        if table_view["community_cards"]
        else "<i>(ще не роздано)</i>"
    )

    lines = [
        "🎲 <b>Таверна: Стіл Техаського Холдему</b>",
        f"<b>Стадія:</b> <code>{stage.upper()}</code> | <b>Банк:</b> 🪙 <b>{pot}</b>",
        f"<b>Спільні карти:</b> {cc}",
        "━━━━━━━━━━━━━━━━━━━",
        "<b>Гравці за столом:</b>",
    ]

    seats = table_view["seats"]
    for idx, s in enumerate(seats):
        if s:
            dealer_mark = " 🔘 (D)" if table_view["dealer_seat"] == idx else ""
            turn_mark = " ⏳ <b>[ХІД]</b>" if table_view["current_turn_seat"] == idx else ""
            bet_info = f" (ставка: {s['current_bet']})" if s["current_bet"] > 0 else ""
            lines.append(
                f"  {idx + 1}. <b>{html.escape(s['display_name'])}</b>: {s['chips']} 🪙{bet_info}{dealer_mark}{turn_mark}"
            )
        else:
            lines.append(f"  {idx + 1}. <i>(Вільне місце)</i>")

    lines.append("━━━━━━━━━━━━━━━━━━━")
    lines.append(f"💬 <i>{html.escape(last_action)}</i>")

    if table_view.get("winners"):
        lines.append("")
        lines.append("🏆 <b>Переможці роздачі:</b>")
        for w in table_view["winners"]:
            lines.append(
                f"  • <b>{html.escape(w['display_name'])}</b> (+{w['amount_won']} 🪙): {w['hand_description']}"
            )

    return "\n".join(lines)


def _build_poker_keyboard(chat_id: int, bot_username: str | None = None) -> InlineKeyboardMarkup:
    public_url = _get_webapp_public_url()
    rows = []

    # 1. Main Action Button: Open WebApp if public URL or direct link configured
    if public_url:
        webapp_url = f"{public_url.rstrip('/')}/poker/index.html?chat_id={chat_id}"
        rows.append(
            [
                InlineKeyboardButton(
                    text="🍺 Відкрити Стіл (Mini App)", web_app=WebAppInfo(url=webapp_url)
                )
            ]
        )
    elif bot_username:
        # Fallback to direct app link
        direct_link = f"https://t.me/{bot_username}/poker?startapp=chat_{chat_id}"
        rows.append([InlineKeyboardButton(text="🍺 Відкрити Стіл (Mini App)", url=direct_link)])

    # 2. In-Chat Quick Action Buttons
    rows.append(
        [
            InlineKeyboardButton(
                text="🪑 Сісти / Встати", callback_data=f"poker:toggle_seat:{chat_id}"
            ),
            InlineKeyboardButton(text="👀 Мої карти", callback_data=f"poker:peek:{chat_id}"),
            InlineKeyboardButton(text="▶️ Роздати", callback_data=f"poker:start:{chat_id}"),
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                text="✅ Чек / Колл", callback_data=f"poker:action:call:{chat_id}"
            ),
            InlineKeyboardButton(text="❌ Пас", callback_data=f"poker:action:fold:{chat_id}"),
            InlineKeyboardButton(text="🔥 Рейз 2x", callback_data=f"poker:action:raise:{chat_id}"),
        ]
    )

    return InlineKeyboardMarkup(inline_keyboard=rows)


@games_router.message(Command("games", "game", "ігри", "таверна"))
async def games_hub_handler(message: Message, bot: Bot) -> None:
    """Tavern Games hub menu."""
    text = (
        "🎲 <b>Ігрова Зала Таверни</b>\n\n"
        "Ласкаво просимо до гральної кімнати! Тут шумить ель, дзвенять монети та чути шелест колоди карт.\n\n"
        "<b>Доступні розваги:</b>\n"
        "• 🃏 <b>Техаський Холдем:</b> Мультиплеєрний стіл на 6 гравців з WebApp графікою або попапами прямо в чаті!\n"
        "• 🎲 <b>Тавернні Кості:</b> Кидки кубиків D&D та перевірки на удачу (<code>/roll 2d20</code>)\n\n"
        "Оберіть гру або введіть команду <code>/poker</code> для негайного старту роздачі!"
    )
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🃏 Запустити Техаський Холдем",
                    callback_data=f"games:launch_poker:{message.chat.id}",
                )
            ],
        ]
    )
    await safe_reply(message, text, reply_markup=kb, parse_mode="HTML")


@games_router.message(Command("poker", "холдем", "покер"))
async def poker_table_handler(message: Message, bot: Bot) -> None:
    """Open or create a poker table in this chat."""
    chat_id = message.chat.id
    table = games_manager.get_or_create_poker_table(chat_id)
    view = table.to_view_dict()

    bot_user = getattr(bot, "_cached_user", None)
    bot_username = getattr(bot_user, "username", None) if bot_user else None

    card_text = _build_poker_card_html(view)
    kb = _build_poker_keyboard(chat_id, bot_username)
    await safe_reply(message, card_text, reply_markup=kb, parse_mode="HTML")


@games_router.callback_query(F.data.startswith("games:launch_poker:"))
async def on_launch_poker_callback(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.data:
        return
    chat_id = int(callback.data.split(":")[-1])
    table = games_manager.get_or_create_poker_table(chat_id)
    view = table.to_view_dict()

    bot_user = getattr(bot, "_cached_user", None)
    bot_username = getattr(bot_user, "username", None) if bot_user else None

    card_text = _build_poker_card_html(view)
    kb = _build_poker_keyboard(chat_id, bot_username)
    if callback.message:
        await callback.message.edit_text(card_text, reply_markup=kb, parse_mode="HTML")
    await callback.answer()


@games_router.callback_query(F.data.startswith("poker:"))
async def on_poker_action_callback(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.data or not callback.from_user:
        return

    parts = callback.data.split(":")
    if len(parts) < 3:
        return

    action = parts[1]
    chat_id = int(parts[-1])
    user_id = callback.from_user.id
    display_name = callback.from_user.first_name
    username = callback.from_user.username

    table = games_manager.get_or_create_poker_table(chat_id)

    # 1. Peek hole cards (show_alert=True popup)
    if action == "peek":
        p = table.get_player(user_id)
        if not p or not p.hole_cards:
            await callback.answer(
                "🃏 Ви зараз не сидите за столом або карти ще не роздані!", show_alert=True
            )
            return

        hand_desc = ""
        cards_str = "  ".join(c.display for c in p.hole_cards)
        if table.community_cards:
            eval_res = evaluate_hand(p.hole_cards + table.community_cards)
            hand_desc = f"\n\nПоточна комбінація:\n🏆 {eval_res.description_uk}"

        alert_text = f"🂠 Ваші кишенькові карти:\n\n{cards_str}{hand_desc}\n\nСтек: {p.chips} 🪙"
        await callback.answer(alert_text, show_alert=True)
        return

    # 2. Toggle seat (sit or stand)
    if action == "toggle_seat":
        existing = table.get_player(user_id)
        if existing:
            ok, msg = table.stand_up(user_id)
        else:
            ok, msg = table.sit_down(user_id, display_name, username)
        await callback.answer(msg, show_alert=not ok)

    # 3. Start hand
    elif action == "start":
        ok, msg = table.start_hand()
        await callback.answer(msg, show_alert=not ok)

    # 4. In-chat betting moves (call, fold, raise)
    elif action == "action":
        move_type = parts[2]
        if move_type == "call":
            to_call = table.highest_bet - (
                table.get_player(user_id).current_bet if table.get_player(user_id) else 0
            )
            move = MoveType.CHECK if to_call == 0 else MoveType.CALL
            ok, msg = table.player_move(user_id, move)
        elif move_type == "fold":
            ok, msg = table.player_move(user_id, MoveType.FOLD)
        elif move_type == "raise":
            target = table.highest_bet + table.min_raise_step
            ok, msg = table.player_move(user_id, MoveType.RAISE, raise_amount=target)
        else:
            ok, msg = False, "Невідомий хід"

        await callback.answer(msg if not ok else f"Хід зроблено: {msg}", show_alert=not ok)

    # Re-render message card in chat
    view = table.to_view_dict()
    bot_user = getattr(bot, "_cached_user", None)
    bot_username = getattr(bot_user, "username", None) if bot_user else None
    card_text = _build_poker_card_html(view)
    kb = _build_poker_keyboard(chat_id, bot_username)

    if callback.message:
        try:
            await callback.message.edit_text(card_text, reply_markup=kb, parse_mode="HTML")
        except Exception:
            pass
