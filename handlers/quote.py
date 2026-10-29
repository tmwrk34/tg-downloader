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


def _generate_quote_commands() -> list[str]:
    """Generates all valid combinations of prefixes, modifiers, and template letters for aiogram Command."""
    mods = ["", "s", "f", "sf", "fs"]
    tpls = ["", "d", "b", "a", "e", "o"]
    cmds: set[str] = set()
    for prefix in ("q", "quote"):
        for m in mods:
            for t in tpls:
                if m or t:
                    cmds.add(f"{prefix}{m}{t}")
                    cmds.add(f"{prefix}{t}{m}")
                else:
                    cmds.add(prefix)
    cyr_mods = ["", "с", "ф", "сф", "фс"]
    cyr_tpls = ["", "д", "б", "а", "е", "о"]
    for prefix in ("цитата", "ц"):
        for m in cyr_mods:
            for t in cyr_tpls:
                if m or t:
                    cmds.add(f"{prefix}_{m}{t}".strip("_"))
                    cmds.add(f"{prefix}_{t}{m}".strip("_"))
                    cmds.add(f"{prefix}{m}{t}")
                    cmds.add(f"{prefix}{t}{m}")
                else:
                    cmds.add(prefix)
    return sorted(cmds)


def parse_quote_command_modifiers(
    cmd: str,
    raw_args: str,
) -> tuple[bool, bool, str | None, str]:
    """
    Parses layout modifiers (seal, dropcap), background template, and strips flags from raw args.
    Returns: (show_seal, show_dropcap, template_name, cleaned_args)
    """
    cmd = (cmd or "q").lower()
    show_seal = True
    show_dropcap = True
    template_name: str | None = None

    # 1. Parse from command name suffix (e.g. "qsfd", "qd", "qsa", "quotesd")
    suffix = ""
    for prefix in ("quote", "цитата", "q", "ц"):
        if cmd.startswith(prefix):
            suffix = cmd[len(prefix) :].lstrip("_")
            break

    for ch in suffix:
        if ch in ("s", "с"):
            show_seal = False
        elif ch in ("f", "ф"):
            show_dropcap = False
        elif ch in ("d", "д"):
            template_name = "Dagger"
        elif ch in ("b", "б"):
            template_name = "Basic"
        elif ch in ("a", "а"):
            template_name = "Arrow"
        elif ch in ("e", "е"):
            template_name = "Ell"
        elif ch in ("o", "о", "p", "п"):
            template_name = "Fired"

    # 2. Parse inline flags from raw_args if provided
    tokens = raw_args.split()
    remaining: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        lt = t.lower()
        if lt in ("-s", "--no-seal", "-безпечатки"):
            show_seal = False
        elif lt in ("-f", "--no-dropcap", "-безбуквиці", "-безбуквицы"):
            show_dropcap = False
        elif lt in ("-sf", "-fs", "-s-f", "-f-s"):
            show_seal = False
            show_dropcap = False
        elif lt in ("-d", "--dagger"):
            template_name = "Dagger"
        elif lt in ("-b", "--basic"):
            template_name = "Basic"
        elif lt in ("-a", "--arrow"):
            template_name = "Arrow"
        elif lt in ("-e", "--ell"):
            template_name = "Ell"
        elif lt in ("-o", "-p", "--fired"):
            template_name = "Fired"
        elif lt in ("-t", "--template") and i + 1 < len(tokens):
            nt = tokens[i + 1].lower()
            if nt in ("d", "dagger", "кинджал"):
                template_name = "Dagger"
                i += 1
            elif nt in ("b", "basic", "базовий"):
                template_name = "Basic"
                i += 1
            elif nt in ("a", "arrow", "стріла"):
                template_name = "Arrow"
                i += 1
            elif nt in ("e", "ell", "кутовий"):
                template_name = "Ell"
                i += 1
            elif nt in ("o", "p", "f", "fired", "обпалений", "вогонь"):
                template_name = "Fired"
                i += 1
            else:
                remaining.append(t)
        else:
            remaining.append(t)
        i += 1

    cleaned_args = " ".join(remaining).strip()
    return show_seal, show_dropcap, template_name, cleaned_args


@quote_router.message(Command(*_generate_quote_commands()))
async def quote_command_handler(
    message: Message, bot: Bot, command: CommandObject | None = None
) -> None:
    """
    Handles quote commands to generate medieval tavern parchment scrolls.
    Supports modifier commands (/qs = no seal, /qf = no dropcap, /qsf = neither),
    background template selection (/qd, /qb, /qa, /qe, /qo), and custom text on replied images.
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

    # Determine layout modifiers and background template
    show_seal, show_dropcap, template_name, raw_args = parse_quote_command_modifiers(cmd, raw_args)

    # Check for author suppression (standalone '-' or trailing dash)
    suppress_author = False
    explicit_author: str | None = None

    raw_args = raw_args.strip()

    if raw_args in ("-", "—", "–", "--"):
        suppress_author = True
        raw_args = ""
    elif raw_args.endswith((" -", " —", " –", " --")):
        for dash in (" -", " —", " –", " --"):
            if raw_args.endswith(dash):
                raw_args = raw_args[: -len(dash)].strip()
                suppress_author = True
                break
    elif raw_args.endswith(("\n-", "\n—", "\n–")):
        for dash in ("\n-", "\n—", "\n–"):
            if raw_args.endswith(dash):
                raw_args = raw_args[: -len(dash)].strip()
                suppress_author = True
                break

    if not suppress_author and raw_args:
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
            if suppress_author:
                author_name = ""
            elif explicit_author:
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
            if suppress_author:
                author_name = ""
            elif explicit_author:
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
            if suppress_author:
                author_name = ""
            elif explicit_author:
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
                "• <b>Лист із зображенням:</b> надішліть <code>/q &lt;текст&gt;</code> у відповідь на картинку — вона стане 3D-печаткою!\n"
                "• <b>З власним текстом:</b> напишіть <code>/quote &lt;текст цитати&gt;</code>.\n\n"
                "<b>🎨 Вибір фону пергаменту (однією літерою):</b>\n"
                "• <code>d</code> — <b>Dagger</b> (кинджал на дерев'яному столі)\n"
                "• <code>b</code> — <b>Basic</b> (класичний пергамент із цвяхами)\n"
                "• <code>a</code> — <b>Arrow</b> (пергамент зі стрілою/списом)\n"
                "• <code>e</code> — <b>Ell</b> (старовинний кутовий сувій)\n"
                "• <code>o</code> — <b>Fired</b> (обпалені вогнем краї)\n\n"
                "<b>⚙️ Керування оформленням (модифікатори):</b>\n"
                "• <code>s</code> — <b>без печатки</b> (наприклад, <code>/qs</code>, <code>/qsd</code>)\n"
                "• <code>f</code> — <b>без буквиці</b> (наприклад, <code>/qf</code>, <code>/qfa</code>)\n"
                "• <code>sf</code> — <b>без печатки та без буквиці</b> (наприклад, <code>/qsf</code>, <code>/qsfd</code>)\n"
                "• <b>Без автора:</b> додайте <code>-</code> в кінці або дайте команду <code>/q -</code>\n\n"
                "<i>💡 Усі літери можна комбінувати:</i> <code>/qd</code> (кинджал), <code>/qsa</code> (стріла без печатки), "
                "<code>/qsfd</code> (чистий текст на кинджалі), <code>/qo</code> (обпалений). Також працюють прапорці: "
                "<code>-d</code>, <code>-b</code>, <code>-a</code>, <code>-e</code>, <code>-o</code>, <code>-t dagger</code>."
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
            template_name=template_name,
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
