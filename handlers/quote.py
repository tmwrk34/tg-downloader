from __future__ import annotations

import io
import logging

from aiogram import Bot, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message

from services.quote_generator import QuoteGeneratorService
from utils.responses import safe_reply

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
    """Exhaustively retrieves avatar photo bytes via profile photos, chat info, or attached media."""
    # 1. Try bot.get_user_profile_photos
    if user_id:
        try:
            photos = await bot.get_user_profile_photos(user_id=user_id, limit=1)
            if photos and photos.photos and photos.photos[0]:
                best_photo = photos.photos[0][-1]
                file_info = await bot.get_file(best_photo.file_id)
                if file_info.file_path:
                    stream = io.BytesIO()
                    await bot.download_file(file_info.file_path, destination=stream)
                    val = stream.getvalue()
                    if val:
                        logger.info(
                            "Downloaded avatar for user %s via profile photos (%d bytes)",
                            user_id,
                            len(val),
                        )
                        return val
        except Exception as e:
            logger.info("Could not fetch profile photos for user %s: %s", user_id, e)

        # 2. Try bot.get_chat(user_id).photo (works even when user restricts profile photos)
        try:
            chat = await bot.get_chat(chat_id=user_id)
            if chat and chat.photo and chat.photo.big_file_id:
                file_info = await bot.get_file(chat.photo.big_file_id)
                if file_info.file_path:
                    stream = io.BytesIO()
                    await bot.download_file(file_info.file_path, destination=stream)
                    val = stream.getvalue()
                    if val:
                        logger.info(
                            "Downloaded avatar for user %s via get_chat.photo (%d bytes)",
                            user_id,
                            len(val),
                        )
                        return val
        except Exception as e:
            logger.info("Could not fetch chat photo for user %s: %s", user_id, e)

    # 3. Check if command message itself has a photo attached
    if message and message.photo:
        try:
            best_photo = message.photo[-1]
            file_info = await bot.get_file(best_photo.file_id)
            if file_info.file_path:
                stream = io.BytesIO()
                await bot.download_file(file_info.file_path, destination=stream)
                val = stream.getvalue()
                if val:
                    logger.info("Using attached photo from command message (%d bytes)", len(val))
                    return val
        except Exception as e:
            logger.info("Could not download photo from message: %s", e)

    # 4. Check if replied message is or has a photo
    if reply and reply.photo:
        try:
            best_photo = reply.photo[-1]
            file_info = await bot.get_file(best_photo.file_id)
            if file_info.file_path:
                stream = io.BytesIO()
                await bot.download_file(file_info.file_path, destination=stream)
                val = stream.getvalue()
                if val:
                    logger.info("Using photo from replied message (%d bytes)", len(val))
                    return val
        except Exception as e:
            logger.info("Could not download photo from replied message: %s", e)

    # 5. Check if forward was from a channel/supergroup with a chat photo
    if reply and reply.forward_from_chat and reply.forward_from_chat.photo:
        try:
            file_info = await bot.get_file(reply.forward_from_chat.photo.big_file_id)
            if file_info.file_path:
                stream = io.BytesIO()
                await bot.download_file(file_info.file_path, destination=stream)
                val = stream.getvalue()
                if val:
                    logger.info("Using channel photo from forward (%d bytes)", len(val))
                    return val
        except Exception as e:
            logger.info("Could not download channel photo: %s", e)

    logger.info("No avatar found for user_id=%s, falling back to noble monogram", user_id)
    return None


@quote_router.message(Command("q", "quote", "цитата"))
async def quote_command_handler(message: Message, bot: Bot) -> None:
    """Handles /q, /quote, and /цитата commands to generate medieval tavern quote scrolls."""
    reply = message.reply_to_message

    target_text = ""
    author_name = ""
    author_user_id: int | None = None

    if reply:
        target_text = reply.text or reply.caption or ""
        if reply.forward_from:
            author_name = reply.forward_from.full_name
            author_user_id = reply.forward_from.id
        elif reply.forward_sender_name:
            author_name = reply.forward_sender_name
        elif reply.from_user:
            author_name = reply.from_user.full_name
            author_user_id = reply.from_user.id
        else:
            author_name = "Мандрівник"
    else:
        # Check command arguments
        cmd_parts = (message.text or "").split(maxsplit=1)
        if len(cmd_parts) > 1 and cmd_parts[1].strip():
            target_text = cmd_parts[1].strip()
            if message.from_user:
                author_name = message.from_user.full_name
                author_user_id = message.from_user.id
            else:
                author_name = "Мандрівник"
        else:
            usage_msg = (
                "📜 <b>Як створити сувій цитати:</b>\n\n"
                "• <b>Відповіддю:</b> надішліть <code>/q</code> або <code>/quote</code> у відповідь на повідомлення іншого мандрівника.\n"
                "• <b>З текстом:</b> напишіть <code>/quote &lt;текст цитати&gt;</code>, щоб закарбувати власні мудрі слова."
            )
            await safe_reply(message, usage_msg, parse_mode="HTML")
            return

    if not target_text.strip():
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

    # Download author avatar for wax signet
    avatar_bytes: bytes | None = await _download_avatar_bytes(
        bot, author_user_id, reply=reply, message=message
    )

    # Render quote card asynchronously in threadpool
    try:
        img_bytes = await quote_service.async_render_quote(
            text=target_text,
            author_name=author_name,
            avatar_bytes=avatar_bytes,
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
