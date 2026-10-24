import asyncio
import logging
import time

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputMediaVideo,
    Message,
)

from config import load_config
from downloader import (
    DownloadedMedia,
    MediaDownloaderCore,
    MediaMetadata,
    PlatformServiceRegistry,
    YouTubeService,
)
from filters import MediaUrlFilter, TriggerPhraseFilter
from middlewares import WhitelistMiddleware

# Payload and runtime safety thresholds
MAX_VIDEO_DURATION_SECONDS = 3 * 3600  # 3 hours cap
DOWNLOAD_TIMEOUT_SECONDS = 300  # 5 minutes download timeout
METADATA_CACHE_TTL_SECONDS = 600  # 10 minutes cache for YouTube metadata

# In-memory metadata cache: video_id -> (timestamp, MediaMetadata)
_youtube_metadata_cache: dict[str, tuple[float, MediaMetadata]] = {}

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


async def send_downloaded_media(
    message: Message,
    downloaded: DownloadedMedia,
    is_audio_only: bool = False,
) -> bool:
    """
    Validate filesize against configured bot payload limits and send the media.
    Returns True if media was successfully sent, False if rejected due to size.
    """
    if downloaded.filesize > config.max_payload_bytes:
        filesize_mb = downloaded.filesize / (1024 * 1024)
        limit_mb = config.max_payload_bytes / (1024 * 1024)
        await message.edit_text(
            f"⚠️ Downloaded file is too large ({filesize_mb:.1f} MB).\n"
            f"Configured bot limit is {limit_mb:.0f} MB."
        )
        return False

    # Gallery / Carousel mode (e.g. Instagram / TikTok albums)
    if downloaded.is_gallery and downloaded.gallery_items:
        items = downloaded.gallery_items
        if len(items) == 1:
            m_type, path = items[0]
            file_input = FSInputFile(path)
            caption = f"🎬 {downloaded.title}" if downloaded.title else ""
            if m_type == "video":
                await message.reply_video(
                    video=file_input, caption=caption, supports_streaming=True
                )
            else:
                await message.reply_photo(photo=file_input, caption=caption)
        else:
            chunks = [items[i : i + 10] for i in range(0, len(items), 10)]
            for chunk_idx, chunk in enumerate(chunks):
                media_group: list[InputMediaPhoto | InputMediaVideo] = []
                for item_idx, (m_type, path) in enumerate(chunk):
                    caption = (
                        f"🎬 {downloaded.title}"
                        if (chunk_idx == 0 and item_idx == 0 and downloaded.title)
                        else None
                    )
                    file_input = FSInputFile(path)
                    if m_type == "video":
                        media_group.append(InputMediaVideo(media=file_input, caption=caption))
                    else:
                        media_group.append(InputMediaPhoto(media=file_input, caption=caption))
                await message.reply_media_group(media=media_group)

        if downloaded.extra_audio_path and downloaded.extra_audio_path.exists():
            audio_input = FSInputFile(downloaded.extra_audio_path)
            await message.reply_audio(
                audio=audio_input,
                title=downloaded.track or downloaded.title,
                performer=downloaded.artist or "Unknown Artist",
            )
        return True

    caption = f"🎬 {downloaded.title}" if downloaded.title else ""
    file_input = FSInputFile(downloaded.filepath)
    ext = downloaded.filepath.suffix.lower()

    if is_audio_only or ext in {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac"}:
        thumb_input = (
            FSInputFile(downloaded.thumbnail_path)
            if downloaded.thumbnail_path and downloaded.thumbnail_path.exists()
            else None
        )
        await message.reply_audio(
            audio=file_input,
            title=downloaded.track or downloaded.title,
            performer=downloaded.artist or "Unknown Artist",
            thumbnail=thumb_input,
            duration=downloaded.duration or 0,
        )
    elif ext in {".jpg", ".jpeg", ".png", ".webp"}:
        await message.reply_photo(photo=file_input, caption=caption)
    else:
        await message.reply_video(
            video=file_input,
            caption=caption,
            duration=downloaded.duration or 0,
            width=downloaded.width or 0,
            height=downloaded.height or 0,
            supports_streaming=True,
        )

    return True


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
        video_id = YouTubeService.extract_video_id(url)
        if video_id:
            status_msg = await message.reply("Fetching YouTube media information... ⏳")
            try:
                meta = await service.extract_metadata(url)
                _youtube_metadata_cache[video_id] = (time.monotonic(), meta)
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
                    if size > config.max_payload_bytes:
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
            sent = await send_downloaded_media(status_msg, downloaded)
            if sent:
                await status_msg.delete()
    except TimeoutError:
        logger.warning("Download timed out for %s: %s", platform, url)
        await status_msg.edit_text(
            f"⏱ Download timed out for {platform.capitalize()} (exceeded 5 minutes).\n"
            "The media stream might be too large or the server connection is slow."
        )
    except Exception as e:
        logger.exception("Failed to process media URL %s", url)
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

    # Pre-flight payload size check using cached metadata if available
    try:
        cached = _youtube_metadata_cache.get(video_id)
        if cached and (time.monotonic() - cached[0]) < METADATA_CACHE_TTL_SECONDS:
            meta = cached[1]
        else:
            meta = await service.extract_metadata(url)
            _youtube_metadata_cache[video_id] = (time.monotonic(), meta)

        estimated_size = service.estimate_format_size(meta, quality)
        if estimated_size and estimated_size > config.max_payload_bytes:
            mb = estimated_size / (1024 * 1024)
            limit_mb = config.max_payload_bytes / (1024 * 1024)
            await callback.answer(
                f"⚠️ {label} is too large (~{mb:.0f}MB > {limit_mb:.0f}MB)",
                show_alert=True,
            )
            if callback.message and isinstance(callback.message, Message):
                await callback.message.edit_text(
                    f"⚠️ <b>{label}</b> is estimated at ~{mb:.1f} MB, which exceeds "
                    f"the bot's configured payload limit ({limit_mb:.0f} MB).\n\n"
                    "Download was aborted to conserve resources.\n"
                    "Please select a lower resolution or download the MP3 audio version.",
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
            if callback.message and isinstance(callback.message, Message):
                sent = await send_downloaded_media(
                    callback.message, downloaded, is_audio_only=(quality == "audio")
                )
                if sent:
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
    if config.local_bot_api_url:
        logger.info(
            "Configuring local Telegram Bot API server at %s (2GB payload mode enabled)",
            config.local_bot_api_url,
        )
        api_server = TelegramAPIServer.from_base(config.local_bot_api_url, is_local=False)
        session = AiohttpSession(api=api_server, timeout=1800.0)
        bot = Bot(token=config.bot_token, session=session)
    else:
        logger.info("Using standard Telegram Cloud API (50MB payload limit)")
        session = AiohttpSession(timeout=300.0)
        bot = Bot(token=config.bot_token, session=session)

    logger.info("Bot successfully started. Press Ctrl+C to stop.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
