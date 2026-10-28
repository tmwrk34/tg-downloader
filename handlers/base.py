from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from filters.triggers import TriggerPhraseFilter
from utils.responses import safe_reply

base_router = Router(name="base")

HELP_MANUAL_HTML = (
    "📖 <b>Bard-Bot Commands & Features Manual</b>\n\n"
    "🎬 <b>Media Downloader:</b>\n"
    "Send any supported link directly into the chat:\n"
    "• <b>TikTok:</b> Videos, short links (<code>vt.tiktok.com</code>), and Photo slideshows (with audio).\n"
    "• <b>Instagram:</b> Reels, Posts, and Carousel albums (<code>/reel/</code>, <code>/p/</code>).\n"
    "• <b>YouTube Shorts:</b> Instant direct video download.\n"
    "• <b>YouTube Videos:</b> Quality selector (1080p, 720p, 360p) or MP3 audio extraction with album art.\n"
    "• <b>Pinterest:</b> Images, video pins, idea pin carousels, and short links (<code>pin.it</code>).\n"
    "• <b>Payload Limit:</b> Up to <b>2GB</b> supported via local Bot API server!\n\n"
    "🎲 <b>Tabletop Dice Roller:</b>\n"
    "• <code>/roll [expression]</code> (aliases: <code>/ролл</code>, <code>/кинь</code>, <code>/кидок</code>) — roll dice publicly (defaults to d20).\n"
    "• <code>/dmroll [expression]</code> (aliases: <code>/дмролл</code>, <code>/дм</code>) — secret roll. In groups, creates an interactive button only you can reveal!\n\n"
    "<b>Notation & Modifiers:</b>\n"
    "• <b>Basic:</b> <code>2d6+3</code>, <code>d20</code>, <code>1d20-2</code>\n"
    "• <b>Cyrillic:</b> <code>2д20</code>, <code>2к20</code>, <code>д20</code>, <code>к20</code>\n"
    "• <b>Text Modifiers:</b> <code>2d6p3</code> / <code>2д6п3</code> (+3), <code>2d6m2</code> / <code>2к6м2</code> (-2)\n"
    "• <b>Keep:</b> <code>4d6k3</code> (keep 3 highest), <code>4d6kl2</code> (keep 2 lowest)\n"
    "• <b>Drop:</b> <code>4d6d2</code> (drop 2 lowest), <code>4d6dh1</code> (drop 1 highest)\n"
    "• <b>Combined:</b> <code>4d6k3+2</code>, <code>4к6к3+2</code>, <code>4d6d2m3</code>\n\n"
    "💬 <b>Natural Language Support:</b>\n"
    "Just type in chat without slash commands:\n"
    "• <code>2d20</code>, <code>d6</code>, <code>2д20</code>, <code>к20</code>, <code>4d6k3+2</code>\n"
    "• <code>roll 2d6</code>, <code>throw d20</code>, <code>roll 20 sided dice</code>, <code>кинь 2д20</code>\n"
    "• <code>dm roll 2d20</code>, <code>hidden roll 4d6k3</code>, <code>дм ролл 2д20</code>\n\n"
    "✨ <b>Inline Mode:</b>\n"
    "Type <code>@bot</code> in any chat to choose preset dice or calculate custom rolls on the fly.\n\n"
    "🎭 <b>Tavern Chatterbot (SLM / Markov):</b>\n"
    "• <code>/chatter</code> (aliases: <code>/бард</code>, <code>/дурачок</code>, <code>/балабол</code>) — Interactive control card with inline buttons.\n"
    "• <code>/chatter on|off</code> — Enable or disable the chatterbot in this chat.\n"
    "• <code>/chatter chance &lt;0-100&gt;</code> — Set random message reply chance (%).\n"
    "• <code>/chatter react &lt;0-100&gt;</code> — Set random emoji reaction chance (%).\n"
    "• <code>/chatter cooldown &lt;sec&gt;</code> — Set anti-spam cooldown in seconds.\n"
    "• <code>/chatter learn on|off</code> — Enable or disable vocabulary learning.\n"
    "• <code>/chatter clear</code> — Wipe this chat's learned vocabulary.\n\n"
    "🤖 <b>Munito AI Assistant:</b>\n"
    "• Tag <code>@bard_taverna_bot &lt;question&gt;</code> or reply to any photo/message to invoke AI.\n"
    "• <code>/ai</code> — Interactive assistant card (status, model switch, reasoning effort, token stats).\n\n"
    "📜 <b>Tavern Quote Scrolls:</b>\n"
    "• <code>/q</code> or <code>/quote</code> (alias: <code>/цитата</code>) — Reply to any message to forge an authentic medieval parchment scroll with illuminated botanical drop-caps and wax signet seal.\n"
    "• <code>/quote &lt;text&gt;</code> — Forge a scroll with your own custom words.\n\n"
    "🏓 <b>Utilities:</b>\n"
    "• <code>/help</code> — Show this manual.\n"
    "• <code>ping</code> — Check bot connection and status."
)


@base_router.message(CommandStart())
async def cmd_start(message: Message, is_admin: bool = False, is_private: bool = False) -> None:
    first_name = message.from_user.first_name if message.from_user else "user"
    status_suffix = " 👑 (Admin/Dev)" if is_admin and is_private else ""

    await message.answer(
        f"Hello, {first_name}! 👋{status_suffix}\n\n"
        "🎬 <b>Media Downloader:</b> Send any supported TikTok, Instagram, YouTube, or Pinterest link.\n"
        "🎲 <b>Dice Roller:</b> Use <code>/roll [expression]</code> (e.g. <code>2d6+3</code>, <code>4d6k3</code>) or type dice in chat!\n"
        "🔒 <b>Secret Rolls:</b> Use <code>/dmroll [expression]</code> for hidden rolls.\n"
        "✨ <b>Inline Mode:</b> Type <code>@bot</code> in any chat to roll dice.\n"
        "🎭 <b>Tavern Chatter:</b> Type <code>/chatter</code> to configure the local bard chatterbot.\n"
        "🤖 <b>AI Assistant:</b> Tag <code>@bot &lt;prompt&gt;</code> or type <code>/ai</code> for settings.\n"
        "📖 <b>Help:</b> Type <code>/help</code> for full documentation and options.",
        parse_mode="HTML",
    )


@base_router.message(Command("help", "хелп", "довідка", "помощь"))
@base_router.message(TriggerPhraseFilter(category="help", exact_match=True))
async def help_command_handler(message: Message) -> None:
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎲 Try Inline Dice",
                    switch_inline_query_current_chat="",
                )
            ]
        ]
    )
    await safe_reply(message, HELP_MANUAL_HTML, reply_markup=keyboard, parse_mode="HTML")


@base_router.message(TriggerPhraseFilter(category="ping", exact_match=True))
async def ping_handler(message: Message, matched_phrase: str = "") -> None:
    await safe_reply(
        message, "Pong! 🏓 Alive, well, and comfortably running inside a Raspberry Pi."
    )
