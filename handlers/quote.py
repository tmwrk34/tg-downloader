from __future__ import annotations

import logging

from aiogram import Bot, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, Message

from services.quote_generator import QuoteGeneratorService
from utils.responses import safe_reply
from utils.telegram_media import download_telegram_file_id

logger = logging.getLogger(__name__)

quote_router = Router(name="quote")

# Global singleton or lazy initialized service
quote_service = QuoteGeneratorService()


async def _download_avatar_bytes(
    bot: Bot,
    user_id: int | None,
    reply: Message | None = None,
    message: Message | None = None,
) -> bytes | None:
    """Exhaustively retrieves photo bytes via replied media, attached photo, profile photos, or chat info."""
    # 1. If replied message is or has a photo, prioritize it for the wax seal
    if reply and reply.photo:
        try:
            best_photo = reply.photo[-1]
            val = await download_telegram_file_id(bot, best_photo.file_id)
            if val:
                logger.info("Using photo from replied message for seal (%d bytes)", len(val))
                return val
        except Exception as e:
            logger.warning("Could not download photo from replied message: %s", e)

    # 2. Check if command message itself has a photo attached
    if message and message.photo:
        try:
            best_photo = message.photo[-1]
            val = await download_telegram_file_id(bot, best_photo.file_id)
            if val:
                logger.info(
                    "Using attached photo from command message for seal (%d bytes)", len(val)
                )
                return val
        except Exception as e:
            logger.warning("Could not download photo from message: %s", e)

    # 3. Check user profile photos
    if user_id:
        try:
            photos = await bot.get_user_profile_photos(user_id=user_id, limit=1)
            if photos and photos.photos and photos.photos[0]:
                best_photo = photos.photos[0][-1]
                val = await download_telegram_file_id(bot, best_photo.file_id)
                if val:
                    logger.info(
                        "Downloaded avatar for user %s via profile photos (%d bytes)",
                        user_id,
                        len(val),
                    )
                    return val
        except Exception as e:
            logger.warning("Could not fetch profile photos for user %s: %s", user_id, e)

        # 4. Try bot.get_chat(user_id).photo (works even when user restricts profile photos)
        try:
            chat = await bot.get_chat(chat_id=user_id)
            if chat and chat.photo and chat.photo.big_file_id:
                val = await download_telegram_file_id(bot, chat.photo.big_file_id)
                if val:
                    logger.info(
                        "Downloaded avatar for user %s via get_chat.photo (%d bytes)",
                        user_id,
                        len(val),
                    )
                    return val
        except Exception as e:
            logger.warning("Could not fetch chat photo for user %s: %s", user_id, e)

    # 5. Check if forward was from a channel/supergroup with a chat photo
    if reply and reply.forward_from_chat and reply.forward_from_chat.photo:
        try:
            val = await download_telegram_file_id(bot, reply.forward_from_chat.photo.big_file_id)
            if val:
                logger.info("Using channel photo from forward (%d bytes)", len(val))
                return val
        except Exception as e:
            logger.warning("Could not download channel photo: %s", e)

    logger.warning("No avatar found for user_id=%s, falling back to noble monogram", user_id)
    return None


@quote_router.message(
    Command(
        "q",
        "qs",
        "qf",
        "qsf",
        "qfs",
        "quote",
        "quotes",
        "quotef",
        "quotesf",
        "quotefs",
        "цитата",
        "цитата_с",
        "цитата_ф",
        "цитата_сф",
        "цитата_фс",
    )
)
async def quote_command_handler(
    message: Message, bot: Bot, command: CommandObject | None = None
) -> None:
    """
    Handles quote commands to generate medieval tavern parchment scrolls.
    Supports modifier commands (/qs = no seal, /qf = no dropcap, /qsf = neither)
    and custom text on replied images.
    """
    reply = message.reply_to_message

    cmd = "q"
    raw_args = ""
    if command is not None:
        cmd = (command.command or "q").lower()
        raw_args = (command.args or "").strip()
    elif message.text:
        parts = message.text.split(maxsplit=1)
        if parts:
            cmd = parts[0].lstrip("/").split("@")[0].lower()
            if len(parts) > 1:
                raw_args = parts[1].strip()

    # Determine layout modifiers based on command alias
    show_seal = True
    show_dropcap = True

    if cmd in ("qs", "quotes", "цитата_с"):
        show_seal = False
    elif cmd in ("qf", "quotef", "цитата_ф"):
        show_dropcap = False
    elif cmd in ("qsf", "qfs", "quotesf", "quotefs", "цитата_сф", "цитата_фс"):
        show_seal = False
        show_dropcap = False

    # Also parse potential inline flags in command arguments
    if raw_args:
        tokens = raw_args.split()
        remaining_tokens: list[str] = []
        for token in tokens:
            lower_t = token.lower()
            if lower_t in ("-s", "--no-seal", "-безпечатки"):
                show_seal = False
            elif lower_t in ("-f", "--no-dropcap", "-безбуквиці", "-безбуквицы"):
                show_dropcap = False
            elif lower_t in ("-sf", "-fs", "-s-f", "-f-s"):
                show_seal = False
                show_dropcap = False
            else:
                remaining_tokens.append(token)
        raw_args = " ".join(remaining_tokens).strip()

    # Parse optional explicit author in args (supports —, –, --, -, or newline dash)
    explicit_author: str | None = None
    for sep in ("\n— ", "\n– ", "\n- ", " — ", " – ", " -- "):
        if sep in raw_args:
            parts = raw_args.rsplit(sep, 1)
            if parts[0].strip() and parts[1].strip():
                raw_args = parts[0].strip()
                explicit_author = parts[1].strip()
                break
    if not explicit_author and " - " in raw_args:
        parts = raw_args.rsplit(" - ", 1)
        candidate = parts[1].strip()
        if (
            parts[0].strip()
            and candidate
            and len(candidate) <= 60
            and not candidate.endswith((".", "!", "?", ";"))
        ):
            raw_args = parts[0].strip()
            explicit_author = candidate

    target_text = ""
    author_name = ""
    author_user_id: int | None = None

    if reply:
        # Replying to another message (or image)
        if raw_args:
            # Custom text provided in command message (e.g. letter over replied image)
            target_text = raw_args
            if explicit_author:
                author_name = explicit_author
            elif reply.photo:
                # User wrote text for a letter with replied photo as seal
                author_name = message.from_user.full_name if message.from_user else "Мандрівник"
                author_user_id = message.from_user.id if message.from_user else None
            elif reply.forward_from:
                author_name = reply.forward_from.full_name
                author_user_id = reply.forward_from.id
            elif reply.from_user:
                author_name = reply.from_user.full_name
                author_user_id = reply.from_user.id
            else:
                author_name = message.from_user.full_name if message.from_user else "Мандрівник"
                author_user_id = message.from_user.id if message.from_user else None
        else:
            # Quoting replied message content
            target_text = reply.text or reply.caption or ""
            if explicit_author:
                author_name = explicit_author
            elif reply.forward_from:
                author_name = reply.forward_from.full_name
                author_user_id = reply.forward_from.id
            elif reply.forward_sender_name:
                author_name = reply.forward_sender_name
                if reply.from_user:
                    author_user_id = reply.from_user.id
            elif reply.from_user:
                author_name = reply.from_user.full_name
                author_user_id = reply.from_user.id
            else:
                author_name = "Мандрівник"
    else:
        # Command used standalone with text argument
        if raw_args:
            target_text = raw_args
            if explicit_author:
                author_name = explicit_author
            elif message.from_user:
                author_name = message.from_user.full_name
                author_user_id = message.from_user.id
            else:
                author_name = "Мандрівник"
        else:
            usage_msg = (
                "📜 <b>Як створити сувій цитати або лист:</b>\n\n"
                "• <b>Відповіддю:</b> надішліть <code>/q</code> у відповідь на повідомлення мандрівника.\n"
                "• <b>Лист із зображенням:</b> надішліть <code>/q &lt;текст&gt;</code> у відповідь на будь-яку картинку — вона стане гербовою печаткою!\n"
                "• <b>Власний текст:</b> напишіть <code>/quote &lt;текст цитати&gt;</code>.\n\n"
                "<b>Керування оформленням:</b>\n"
                "• <code>/qs</code> — створити сувій <b>без печатки</b>\n"
                "• <code>/qf</code> — створити сувій <b>без буквиці</b> (звичайний рукописний текст)\n"
                "• <code>/qsf</code> або <code>/qfs</code> — сувій <b>без печатки та без буквиці</b>"
            )
            await safe_reply(message, usage_msg, parse_mode="HTML")
            return

    if not target_text.strip():
        if reply and reply.photo:
            hint_msg = (
                "📜 <b>Лист із зображенням на печатці:</b>\n"
                f"Надішліть <code>/{cmd} &lt;текст листа&gt;</code> у відповідь на цю картинку, "
                "щоб створити сувій із печаткою з цього зображення."
            )
            await safe_reply(message, hint_msg, parse_mode="HTML")
        else:
            await safe_reply(
                message,
                "⚠️ Повідомлення не містить тексту для створення цитати.",
                parse_mode="HTML",
            )
        return

    # Indicate activity
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.UPLOAD_PHOTO)
    except Exception as e:
        logger.debug("Failed sending chat action: %s", e)

    # Download image for wax seal (replied photo > command photo > user avatar)
    avatar_bytes: bytes | None = None
    if show_seal:
        avatar_bytes = await _download_avatar_bytes(
            bot, author_user_id, reply=reply, message=message
        )

    # Render quote card asynchronously in threadpool
    try:
        img_bytes = await quote_service.async_render_quote(
            text=target_text,
            author_name=author_name,
            avatar_bytes=avatar_bytes,
            show_seal=show_seal,
            show_dropcap=show_dropcap,
        )

        photo = BufferedInputFile(img_bytes, filename="quote.jpg")
        reply_to_id = reply.message_id if reply else message.message_id

        await bot.send_photo(
            chat_id=message.chat.id,
            photo=photo,
            reply_to_message_id=reply_to_id,
        )
    except Exception as e:
        logger.exception("Error generating quote card: %s", e)
        await safe_reply(
            message,
            "⚠️ Не вдалося закарбувати сувій цитати. Спробуйте пізніше.",
            parse_mode="HTML",
        )
