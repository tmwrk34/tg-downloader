import asyncio
import logging
import re

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from config import load_config
from downloader import MediaDownloaderCore, PlatformServiceRegistry, YouTubeService
from filters import MediaUrlFilter, TriggerPhraseFilter
from middlewares import WhitelistMiddleware

# Payload and runtime safety thresholds
MAX_VIDEO_DURATION_SECONDS = 3 * 3600  # 3 hours cap
MAX_STD_BOT_API_PAYLOAD_BYTES = 50 * 1024 * 1024  # 50 MB standard Telegram Bot API limit
DOWNLOAD_TIMEOUT_SECONDS = 300  # 5 minutes download timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose event logs for every handled/unhandled update
logging.getLogger("aiogram.event").setLevel(logging.WARNING)

# Load and validate configuration
config = load_config()

# Initialize dispatcher and core services
dp = Dispatcher()
downloader_core = MediaDownloaderCore(download_dir="./downloads")
platform_registry = PlatformServiceRegistry(downloader_core)

# Register WhitelistMiddleware at the outer layer
dp.message.outer_middleware(
    WhitelistMiddleware(
        allowed_chat_ids=config.allowed_chat_ids,
        admin_user_ids=config.admin_user_ids,
    )
)
dp.callback_query.outer_middleware(
    WhitelistMiddleware(
        allowed_chat_ids=config.allowed_chat_ids,
        admin_user_ids=config.admin_user_ids,
    )
)


def format_duration(seconds: int | None) -> str:
    """Format duration in seconds to HH:MM:SS or MM:SS."""
    if not seconds:
        return "N/A"
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


@dp.message(CommandStart())
async def cmd_start(message: Message, is_admin: bool = False, is_private: bool = False):
    first_name = message.from_user.first_name if message.from_user else "user"
    status_suffix = " 👑 (Admin/Dev)" if is_admin and is_private else ""

    await message.answer(
        f"Hello, {first_name}! 👋{status_suffix}\n"
        f"I am a media downloader bot. Send me a link to get started!"
    )


@dp.message(TriggerPhraseFilter(category="ping", exact_match=True))
async def ping_handler(message: Message, matched_phrase: str = ""):
    await message.reply("Pong! 🏓 Alive, well, and comfortably running inside a Raspberry Pi.")


@dp.message(MediaUrlFilter())
async def media_url_handler(message: Message, url: str, platform: str):
    logger.info("Matched %s URL: %s", platform, url)

    service = platform_registry.get_service_by_name(
        platform
    ) or platform_registry.get_service_for_url(url)
    if not service:
        await message.reply("Unsupported platform.")
        return

    # For YouTube: prompt user with resolution and audio options via inline keyboard
    if platform == "youtube" and isinstance(service, YouTubeService):
        vid_match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([a-zA-Z0-9_-]{11})", url)
        if vid_match:
            video_id = vid_match.group(1)
            status_msg = await message.reply("Fetching YouTube media information... ⏳")
            try:
                meta = await service.extract_metadata(url)
                dur_str = format_duration(meta.duration)

                # Cap video duration at 3 hours
                if meta.duration and meta.duration > MAX_VIDEO_DURATION_SECONDS:
                    await status_msg.edit_text(
                        f"⚠️ <b>Video is too long:</b> {dur_str}\n"
                        "Maximum allowed duration is 3 hours to prevent server exhaustion.",
                        parse_mode="HTML",
                    )
                    return

                size_1080 = service.estimate_format_size(meta, "1080")
                size_720 = service.estimate_format_size(meta, "720")
                size_360 = service.estimate_format_size(meta, "360")
                size_audio = service.estimate_format_size(meta, "audio")

                def make_btn_label(quality: str, size: int | None) -> str:
                    base = {
                        "1080": "🎬 1080p MP4",
                        "720": "🎬 720p MP4",
                        "360": "🎬 360p MP4",
                        "audio": "🎵 Audio MP3",
                    }[quality]
                    if not size:
                        return base
                    mb = size / (1024 * 1024)
                    if size > MAX_STD_BOT_API_PAYLOAD_BYTES:
                        return f"{base} (~{mb:.0f}MB ⚠️)"
                    return f"{base} (~{mb:.0f}MB)"

                text = (
                    f"🎬 <b>{meta.title}</b>\n"
                    f"⏱ <b>Duration:</b> {dur_str}\n\n"
                    "Select download format / quality:"
                )
                keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text=make_btn_label("1080", size_1080),
                                callback_data=f"yt:1080:{video_id}",
                            ),
                            InlineKeyboardButton(
                                text=make_btn_label("720", size_720),
                                callback_data=f"yt:720:{video_id}",
                            ),
                        ],
                        [
                            InlineKeyboardButton(
                                text=make_btn_label("360", size_360),
                                callback_data=f"yt:360:{video_id}",
                            ),
                            InlineKeyboardButton(
                                text=make_btn_label("audio", size_audio),
                                callback_data=f"yt:audio:{video_id}",
                            ),
                        ],
                    ]
                )
                await status_msg.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
                return
            except Exception as e:
                logger.exception("Failed to fetch YouTube info for %s", url)
                await status_msg.edit_text(f"Failed to fetch YouTube metadata: {e}")
                return

    # For TikTok, Instagram, and direct platforms: download immediately
    status_msg = await message.reply(f"Downloading {platform.capitalize()} media... Please wait ⏳")
    try:
        downloaded = await asyncio.wait_for(
            service.download(url),
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
        async with downloaded:
            if downloaded.filesize > MAX_STD_BOT_API_PAYLOAD_BYTES:
                filesize_mb = downloaded.filesize / 1024 / 1024
                await status_msg.edit_text(
                    f"⚠️ Downloaded file is too large ({filesize_mb:.1f} MB).\n"
                    "Standard Telegram Bot API limits uploads to 50 MB.\n"
                    "Full files up to 2GB will be available once Local Bot API (#18) is deployed!"
                )
                return

            caption = f"🎬 {downloaded.title}" if downloaded.title else ""
            file_input = FSInputFile(downloaded.filepath)
            ext = downloaded.filepath.suffix.lower()

            if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                await message.reply_photo(photo=file_input, caption=caption)
            elif ext in {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac"}:
                thumb_input = (
                    FSInputFile(downloaded.thumbnail_path)
                    if downloaded.thumbnail_path and downloaded.thumbnail_path.exists()
                    else None
                )
                await message.reply_audio(
                    audio=file_input,
                    caption=caption,
                    title=downloaded.track or downloaded.title,
                    performer=downloaded.artist or "Unknown Artist",
                    thumbnail=thumb_input,
                    duration=downloaded.duration or 0,
                )
            else:
                await message.reply_video(
                    video=file_input,
                    caption=caption,
                    duration=downloaded.duration or 0,
                    width=downloaded.width or 0,
                    height=downloaded.height or 0,
                    supports_streaming=True,
                )

            await status_msg.delete()
    except TimeoutError:
        logger.warning("Download timed out for %s: %s", platform, url)
        await status_msg.edit_text(
            f"⏱ Download timed out for {platform.capitalize()} (exceeded 5 minutes).\n"
            "The media stream might be too large or the server connection is slow."
        )
    except Exception as e:
        logger.exception("Failed to process media URL %s", url)
        err_str = str(e)
        if "No video formats found" in err_str:
            await status_msg.edit_text(
                "ℹ️ This post contains a photo gallery without a video stream.\n"
                "Multi-image photo album support is coming soon in Issue #9!"
            )
        else:
            await status_msg.edit_text(f"Failed to download media: {e}")


@dp.callback_query(F.data.startswith("yt:"))
async def youtube_callback_handler(callback: CallbackQuery):
    if not callback.data:
        return

    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer("Invalid callback query", show_alert=True)
        return

    _, quality, video_id = parts
    url = f"https://www.youtube.com/watch?v={video_id}"
    service = platform_registry.get_service_by_name("youtube")
    if not service or not isinstance(service, YouTubeService):
        await callback.answer("YouTube service unavailable", show_alert=True)
        return

    quality_labels = {
        "1080": "1080p MP4",
        "720": "720p MP4",
        "360": "360p MP4",
        "audio": "Audio MP3",
    }
    label = quality_labels.get(quality, quality)

    # Pre-flight payload size check before starting download
    try:
        meta = await service.extract_metadata(url)
        estimated_size = service.estimate_format_size(meta, quality)
        if estimated_size and estimated_size > MAX_STD_BOT_API_PAYLOAD_BYTES:
            mb = estimated_size / (1024 * 1024)
            await callback.answer(f"⚠️ {label} is too large (~{mb:.0f} MB > 50 MB)", show_alert=True)
            if callback.message and isinstance(callback.message, Message):
                await callback.message.edit_text(
                    f"⚠️ <b>{label}</b> is estimated at ~{mb:.1f} MB, which exceeds Telegram's 50MB limit.\n\n"
                    "Download was aborted to save server bandwidth.\n"
                    "Please select 🎬 360p or 🎵 Audio MP3 until Local Bot API (#18) is deployed!",
                    parse_mode="HTML",
                )
            return
    except Exception as e:
        logger.warning("Could not pre-estimate size for %s: %s", url, e)

    await callback.answer()
    if callback.message and isinstance(callback.message, Message):
        await callback.message.edit_text(f"Downloading YouTube {label}... Please wait ⏳")

    try:
        format_opts = service.get_format_options(quality)
        downloaded = await asyncio.wait_for(
            service.download(url, extra_opts=format_opts),
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
        async with downloaded:
            if downloaded.filesize > MAX_STD_BOT_API_PAYLOAD_BYTES:
                filesize_mb = downloaded.filesize / 1024 / 1024
                if callback.message and isinstance(callback.message, Message):
                    await callback.message.edit_text(
                        f"⚠️ Downloaded {label} is too large ({filesize_mb:.1f} MB).\n"
                        "Standard Telegram Bot API limits uploads to 50 MB.\n"
                        "Please choose 🎵 Audio MP3 or wait until Local Bot API (#18) is set up!"
                    )
                return

            caption = f"🎬 {downloaded.title}" if downloaded.title else ""
            file_input = FSInputFile(downloaded.filepath)

            if quality == "audio":
                if callback.message and isinstance(callback.message, Message):
                    thumb_input = (
                        FSInputFile(downloaded.thumbnail_path)
                        if downloaded.thumbnail_path and downloaded.thumbnail_path.exists()
                        else None
                    )
                    await callback.message.reply_audio(
                        audio=file_input,
                        title=downloaded.track or downloaded.title,
                        performer=downloaded.artist or "Unknown Artist",
                        thumbnail=thumb_input,
                        duration=downloaded.duration or 0,
                    )
            else:
                if callback.message and isinstance(callback.message, Message):
                    await callback.message.reply_video(
                        video=file_input,
                        caption=caption,
                        duration=downloaded.duration or 0,
                        width=downloaded.width or 0,
                        height=downloaded.height or 0,
                        supports_streaming=True,
                    )

            if callback.message and isinstance(callback.message, Message):
                await callback.message.delete()
    except TimeoutError:
        logger.warning("Download timed out for YouTube %s (%s)", quality, video_id)
        if callback.message and isinstance(callback.message, Message):
            await callback.message.edit_text(
                f"⏱ Download timed out for {label} (exceeded 5 minutes).\n"
                "The video stream may be too large or the server connection is slow."
            )
    except Exception as e:
        logger.exception("Failed to download YouTube %s for %s", quality, video_id)
        if callback.message and isinstance(callback.message, Message):
            await callback.message.edit_text(f"Failed to download YouTube media: {e}")


async def main():
    bot = Bot(token=config.bot_token)
    logger.info("Bot successfully started. Press Ctrl+C to stop.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
