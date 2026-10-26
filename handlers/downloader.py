from __future__ import annotations

import asyncio
import logging
import time

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputMediaVideo,
    InputRichBlockPhoto,
    InputRichBlockSlideshow,
    InputRichBlockUnion,
    InputRichBlockVideo,
    InputRichMessage,
    Message,
    RichBlockCaption,
)

from downloader import DownloadedMedia, MediaMetadata, YouTubeService
from filters.media import MediaUrlFilter
from services.cache import CachedMedia, get_canonical_cache_key
from services.worker import chat_action_heartbeat, set_message_reaction
from utils.proxy import bot_proxy as bot

logger = logging.getLogger(__name__)

downloader_router = Router(name="downloader")

# Payload and runtime safety thresholds
MAX_VIDEO_DURATION_SECONDS = 3 * 3600  # 3 hours cap
DOWNLOAD_TIMEOUT_SECONDS = 300  # 5 minutes download timeout
METADATA_CACHE_TTL_SECONDS = 600  # 10 minutes cache for YouTube metadata
MAX_METADATA_CACHE_SIZE = 100  # Upper bound on cached YouTube metadata entries

# In-memory metadata cache: video_id -> (timestamp, MediaMetadata)
_youtube_metadata_cache: dict[str, tuple[float, MediaMetadata]] = {}


def _cache_youtube_metadata(video_id: str, meta: MediaMetadata) -> None:
    """Store YouTube metadata with TTL pruning and strict upper-bound eviction."""
    now = time.monotonic()
    expired = [
        k
        for k, (ts, _) in _youtube_metadata_cache.items()
        if (now - ts) >= METADATA_CACHE_TTL_SECONDS
    ]
    for k in expired:
        _youtube_metadata_cache.pop(k, None)

    while len(_youtube_metadata_cache) >= MAX_METADATA_CACHE_SIZE:
        oldest_key = next(iter(_youtube_metadata_cache))
        _youtube_metadata_cache.pop(oldest_key, None)

    _youtube_metadata_cache[video_id] = (now, meta)


def format_duration(seconds: int | float | None) -> str:
    """Format duration in seconds to HH:MM:SS or MM:SS."""
    if not seconds:
        return "N/A"
    sec = int(round(seconds))
    mins, secs = divmod(sec, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


async def send_cached_media(message: Message, cached: CachedMedia) -> bool:
    """
    Deliver media instantly using Telegram file_id from cache.
    Raises TelegramAPIError if the file_id is invalid or rejected by Telegram.
    """
    if cached.media_type == "video":
        if not cached.file_id:
            return False
        await message.reply_video(
            video=cached.file_id,
            caption=cached.caption,
            duration=int(cached.extra_data.get("duration", 0)),
            width=int(cached.extra_data.get("width", 0)),
            height=int(cached.extra_data.get("height", 0)),
            supports_streaming=True,
        )
        return True

    elif cached.media_type == "photo":
        if not cached.file_id:
            return False
        await message.reply_photo(
            photo=cached.file_id,
            caption=cached.caption,
        )
        return True

    elif cached.media_type == "audio":
        if not cached.file_id:
            return False
        await message.reply_audio(
            audio=cached.file_id,
            title=cached.extra_data.get("title"),
            performer=cached.extra_data.get("performer"),
            duration=int(cached.extra_data.get("duration", 0)),
        )
        return True

    elif cached.media_type == "album":
        items = cached.extra_data.get("items", [])
        if not items:
            return False

        if len(items) == 1:
            it = items[0]
            if it["type"] == "video":
                await message.reply_video(
                    video=it["file_id"],
                    caption=cached.caption,
                    supports_streaming=True,
                )
            else:
                await message.reply_photo(
                    photo=it["file_id"],
                    caption=cached.caption,
                )
        else:
            chunks = [items[i : i + 10] for i in range(0, len(items), 10)]
            for chunk_idx, chunk in enumerate(chunks):
                caption_str = cached.caption if chunk_idx == 0 else None
                media_group: list[InputMediaPhoto | InputMediaVideo] = []
                for item_idx, it in enumerate(chunk):
                    c = caption_str if item_idx == 0 else None
                    if it["type"] == "video":
                        media_group.append(InputMediaVideo(media=it["file_id"], caption=c))
                    else:
                        media_group.append(InputMediaPhoto(media=it["file_id"], caption=c))
                await message.reply_media_group(media=media_group)

        audio_file_id = cached.extra_data.get("audio_file_id")
        if audio_file_id:
            await message.reply_audio(
                audio=audio_file_id,
                title=cached.extra_data.get("audio_title"),
                performer=cached.extra_data.get("audio_performer", "Unknown Artist"),
            )
        return True

    return False


async def send_downloaded_media(
    message: Message,
    downloaded: DownloadedMedia,
    is_audio_only: bool = False,
    cache_key: str | None = None,
) -> bool:
    """
    Validate filesize against configured bot payload limits and send the media.
    Optionally saves the sent file_id to the media cache if cache_key is provided.
    Returns True if media was successfully sent, False if rejected due to size.
    """
    if downloaded.filesize > bot.config.max_payload_bytes:
        filesize_mb = downloaded.filesize / (1024 * 1024)
        limit_mb = bot.config.max_payload_bytes / (1024 * 1024)
        await message.edit_text(
            f"⚠️ Downloaded file is too large ({filesize_mb:.1f} MB).\n"
            f"Configured bot limit is {limit_mb:.0f} MB."
        )
        return False

    # Gallery / Carousel mode (e.g. Instagram / TikTok albums)
    if downloaded.is_gallery and downloaded.gallery_items:
        items = downloaded.gallery_items
        album_items: list[dict[str, str]] = []
        if len(items) == 1:
            m_type, path = items[0]
            file_input = FSInputFile(path)
            caption = f"🎬 {downloaded.title}" if downloaded.title else ""
            if m_type == "video":
                sent_msg = await message.reply_video(
                    video=file_input, caption=caption, supports_streaming=True
                )
                vid = getattr(sent_msg, "video", None)
                if vid and getattr(vid, "file_id", None):
                    album_items.append({"type": "video", "file_id": vid.file_id})
            else:
                sent_msg = await message.reply_photo(photo=file_input, caption=caption)
                photos = getattr(sent_msg, "photo", None)
                if photos and len(photos) > 0:
                    album_items.append({"type": "photo", "file_id": photos[-1].file_id})
        else:
            chunks = [items[i : i + 10] for i in range(0, len(items), 10)]
            for chunk_idx, chunk in enumerate(chunks):
                caption_str = (
                    f"🎬 {downloaded.title}" if (chunk_idx == 0 and downloaded.title) else None
                )
                sent_rich = False
                try:
                    slides: list[InputRichBlockUnion] = []
                    for m_type, path in chunk:
                        file_input = FSInputFile(path)
                        if m_type == "video":
                            slides.append(
                                InputRichBlockVideo(video=InputMediaVideo(media=file_input))
                            )
                        else:
                            slides.append(
                                InputRichBlockPhoto(photo=InputMediaPhoto(media=file_input))
                            )

                    caption_block = RichBlockCaption(text=caption_str) if caption_str else None
                    slideshow = InputRichBlockSlideshow(blocks=slides, caption=caption_block)
                    await message.reply_rich(rich_message=InputRichMessage(blocks=[slideshow]))
                    sent_rich = True
                except Exception as e:
                    logger.warning(
                        "Failed to send rich slideshow (%s), falling back to media group: %s",
                        type(e).__name__,
                        e,
                    )

                if not sent_rich:
                    media_group: list[InputMediaPhoto | InputMediaVideo] = []
                    for item_idx, (m_type, path) in enumerate(chunk):
                        item_caption = caption_str if item_idx == 0 else None
                        file_input = FSInputFile(path)
                        if m_type == "video":
                            media_group.append(
                                InputMediaVideo(media=file_input, caption=item_caption)
                            )
                        else:
                            media_group.append(
                                InputMediaPhoto(media=file_input, caption=item_caption)
                            )
                    sent_msgs = await message.reply_media_group(media=media_group)
                    if sent_msgs and isinstance(sent_msgs, list):
                        for sm in sent_msgs:
                            vid = getattr(sm, "video", None)
                            photos = getattr(sm, "photo", None)
                            if vid and getattr(vid, "file_id", None):
                                album_items.append({"type": "video", "file_id": vid.file_id})
                            elif photos and len(photos) > 0:
                                album_items.append({"type": "photo", "file_id": photos[-1].file_id})

        audio_file_id = None
        if downloaded.extra_audio_path and downloaded.extra_audio_path.exists():
            audio_input = FSInputFile(downloaded.extra_audio_path)
            sent_audio = await message.reply_audio(
                audio=audio_input,
                title=downloaded.track or downloaded.title,
                performer=downloaded.artist or "Unknown Artist",
            )
            aud = getattr(sent_audio, "audio", None)
            if aud and getattr(aud, "file_id", None):
                audio_file_id = aud.file_id

        if cache_key and album_items:
            try:
                await bot.media_cache.set(
                    cache_key=cache_key,
                    media_type="album",
                    file_id=None,
                    caption=f"🎬 {downloaded.title}" if downloaded.title else "",
                    extra_data={
                        "items": album_items,
                        "audio_file_id": audio_file_id,
                        "audio_title": downloaded.track or downloaded.title,
                        "audio_performer": downloaded.artist or "Unknown Artist",
                    },
                )
            except Exception as e:
                logger.warning("Failed to cache gallery for %s: %s", cache_key, e)
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
        sent_audio = await message.reply_audio(
            audio=file_input,
            title=downloaded.track or downloaded.title,
            performer=downloaded.artist or "Unknown Artist",
            thumbnail=thumb_input,
            duration=int(round(downloaded.duration)) if downloaded.duration else 0,
        )
        aud = getattr(sent_audio, "audio", None)
        if cache_key and aud and getattr(aud, "file_id", None):
            try:
                await bot.media_cache.set(
                    cache_key=cache_key,
                    media_type="audio",
                    file_id=aud.file_id,
                    caption="",
                    extra_data={
                        "title": downloaded.track or downloaded.title,
                        "performer": downloaded.artist or "Unknown Artist",
                        "duration": int(round(downloaded.duration)) if downloaded.duration else 0,
                    },
                )
            except Exception as e:
                logger.warning("Failed to cache audio for %s: %s", cache_key, e)
    elif ext in {".jpg", ".jpeg", ".png", ".webp"}:
        sent_photo = await message.reply_photo(photo=file_input, caption=caption)
        photos = getattr(sent_photo, "photo", None)
        if cache_key and photos and len(photos) > 0:
            try:
                await bot.media_cache.set(
                    cache_key=cache_key,
                    media_type="photo",
                    file_id=photos[-1].file_id,
                    caption=caption,
                    extra_data={},
                )
            except Exception as e:
                logger.warning("Failed to cache photo for %s: %s", cache_key, e)
    else:
        sent_video = await message.reply_video(
            video=file_input,
            caption=caption,
            duration=int(round(downloaded.duration)) if downloaded.duration else 0,
            width=int(round(downloaded.width)) if downloaded.width else 0,
            height=int(round(downloaded.height)) if downloaded.height else 0,
            supports_streaming=True,
        )
        vid = getattr(sent_video, "video", None)
        if cache_key and vid and getattr(vid, "file_id", None):
            try:
                await bot.media_cache.set(
                    cache_key=cache_key,
                    media_type="video",
                    file_id=vid.file_id,
                    caption=caption,
                    extra_data={
                        "duration": int(round(downloaded.duration)) if downloaded.duration else 0,
                        "width": int(round(downloaded.width)) if downloaded.width else 0,
                        "height": int(round(downloaded.height)) if downloaded.height else 0,
                    },
                )
            except Exception as e:
                logger.warning("Failed to cache video for %s: %s", cache_key, e)

    return True


@downloader_router.message(MediaUrlFilter())
async def media_url_handler(message: Message, url: str, platform: str) -> None:
    logger.info("Matched %s URL: %s", platform, url)

    service = bot.platform_registry.get_service_by_name(
        platform
    ) or bot.platform_registry.get_service_for_url(url)
    if not service:
        await message.reply("Unsupported platform.")
        return

    # Check cache for immediate delivery (TikTok, Instagram, YouTube Shorts, etc.)
    is_yt_long = (
        platform == "youtube"
        and isinstance(service, YouTubeService)
        and not YouTubeService.is_short(url)
    )

    cache_key: str | None = None
    if not is_yt_long:
        cache_key = await get_canonical_cache_key(url, platform=platform)
        cached = await bot.media_cache.get(cache_key)
        if cached:
            try:
                sent = await bot.send_cached_media(message, cached)
                if sent:
                    logger.info("Cache HIT: Delivered %s (%s)", platform, cache_key)
                    return
            except TelegramAPIError as e:
                logger.warning(
                    "Cached media send failed for %s (%s), purging cache and downloading: %s",
                    cache_key,
                    type(e).__name__,
                    e,
                )
                await bot.media_cache.delete(cache_key)

    # For YouTube: prompt user with resolution and audio options via inline keyboard
    if is_yt_long:
        video_id = YouTubeService.extract_video_id(url)
        if video_id:
            status_msg = await message.reply("Fetching YouTube media information... ⏳")
            try:
                cached_meta = _youtube_metadata_cache.get(video_id)
                if cached_meta and (time.monotonic() - cached_meta[0]) < METADATA_CACHE_TTL_SECONDS:
                    meta = cached_meta[1]
                else:
                    meta = await service.extract_metadata(url)
                    _cache_youtube_metadata(video_id, meta)
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
                    if size > bot.config.max_payload_bytes:
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

    # For TikTok, Instagram, YouTube Shorts, and direct platforms: download immediately
    label = (
        "YouTube Short"
        if (platform == "youtube" and YouTubeService.is_short(url))
        else platform.capitalize()
    )

    async with bot.worker_limiter.singleflight(cache_key):
        if cache_key:
            cached = await bot.media_cache.get(cache_key)
            if cached:
                try:
                    sent = await bot.send_cached_media(message, cached)
                    if sent:
                        logger.info(
                            "Cache HIT (after singleflight): Delivered %s (%s)", platform, cache_key
                        )
                        return
                except TelegramAPIError:
                    await bot.media_cache.delete(cache_key)

        initial_text = (
            f"⏳ Queued {label} media for download... Please wait"
            if bot.worker_limiter.is_download_busy
            else f"Downloading {label} media... Please wait ⏳"
        )
        status_msg = await message.reply(initial_text)
        was_queued = False

        async def _on_queued(pos: int) -> None:
            nonlocal was_queued
            was_queued = True
            await set_message_reaction(message, "⏳")
            await status_msg.edit_text(
                f"⏳ Queued {label} download (position #{pos})... Please wait"
            )

        async def _on_active() -> None:
            await set_message_reaction(message, "⚡")
            if was_queued:
                await status_msg.edit_text(f"Downloading {label} media... Please wait ⏳")

        try:
            async with (
                bot.worker_limiter.acquire_download(on_queued=_on_queued, on_active=_on_active),
                chat_action_heartbeat(message, ChatAction.UPLOAD_VIDEO),
            ):
                downloaded = await asyncio.wait_for(
                    service.download(url),
                    timeout=DOWNLOAD_TIMEOUT_SECONDS,
                )
                async with downloaded:
                    sent = await bot.send_downloaded_media(
                        status_msg, downloaded, cache_key=cache_key
                    )
                    if sent:
                        await status_msg.delete()
                        await set_message_reaction(message, None)
                    else:
                        await set_message_reaction(message, "❌")
        except TimeoutError:
            logger.warning("Download timed out for %s: %s", platform, url)
            await set_message_reaction(message, "❌")
            await status_msg.edit_text(
                f"⏱ Download timed out for {platform.capitalize()} (exceeded 5 minutes).\n"
                "The media stream might be too large or the server connection is slow."
            )
        except Exception as e:
            logger.exception("Failed to process media URL %s", url)
            await set_message_reaction(message, "❌")
            await status_msg.edit_text(f"Failed to download media: {e}")


@downloader_router.callback_query(F.data.startswith("yt:"))
async def youtube_callback_handler(callback: CallbackQuery) -> None:
    if not callback.data:
        return

    parts = callback.data.split(":")
    if len(parts) != 3:
        await callback.answer("Invalid callback query", show_alert=True)
        return

    _, quality, video_id = parts
    cache_key = f"youtube:{video_id}:{quality}"
    cached = await bot.media_cache.get(cache_key)
    if cached:
        try:
            if callback.message and isinstance(callback.message, Message):
                sent = await bot.send_cached_media(callback.message, cached)
                if sent:
                    await callback.message.delete()
                    logger.info("Cache HIT: Delivered YouTube %s (%s)", quality, cache_key)
                    return
        except TelegramAPIError as e:
            logger.warning(
                "Cached YouTube file_id failed (%s), invalidating cache: %s",
                type(e).__name__,
                e,
            )
            await bot.media_cache.delete(cache_key)

    url = f"https://www.youtube.com/watch?v={video_id}"
    service = bot.platform_registry.get_service_by_name("youtube")
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
        cached_meta = _youtube_metadata_cache.get(video_id)
        if cached_meta and (time.monotonic() - cached_meta[0]) < METADATA_CACHE_TTL_SECONDS:
            meta = cached_meta[1]
        else:
            meta = await service.extract_metadata(url)
            _cache_youtube_metadata(video_id, meta)

        estimated_size = service.estimate_format_size(meta, quality)
        if estimated_size and estimated_size > bot.config.max_payload_bytes:
            mb = estimated_size / (1024 * 1024)
            limit_mb = bot.config.max_payload_bytes / (1024 * 1024)
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
    target_msg = callback.message if isinstance(callback.message, Message) else None

    async with bot.worker_limiter.singleflight(cache_key):
        cached = await bot.media_cache.get(cache_key)
        if cached and target_msg:
            try:
                sent = await bot.send_cached_media(target_msg, cached)
                if sent:
                    await target_msg.delete()
                    logger.info(
                        "Cache HIT (after singleflight): Delivered YouTube %s (%s)",
                        quality,
                        cache_key,
                    )
                    return
            except TelegramAPIError:
                await bot.media_cache.delete(cache_key)

        was_queued = False

        async def _on_yt_queued(pos: int) -> None:
            nonlocal was_queued
            was_queued = True
            if target_msg:
                await set_message_reaction(target_msg, "⏳")
                await target_msg.edit_text(
                    f"⏳ Queued YouTube {label} (position #{pos})... Please wait"
                )

        async def _on_yt_active() -> None:
            if target_msg:
                await set_message_reaction(target_msg, "⚡")
                await target_msg.edit_text(f"Downloading YouTube {label}... Please wait ⏳")

        if target_msg and not bot.worker_limiter.is_download_busy:
            await target_msg.edit_text(f"Downloading YouTube {label}... Please wait ⏳")

        chat_action = ChatAction.UPLOAD_VOICE if quality == "audio" else ChatAction.UPLOAD_VIDEO

        try:
            async with bot.worker_limiter.acquire_download(
                on_queued=_on_yt_queued, on_active=_on_yt_active
            ):
                format_opts = service.get_format_options(quality)
                if target_msg:
                    async with chat_action_heartbeat(target_msg, chat_action):
                        if quality == "audio":
                            async with bot.worker_limiter.acquire_transcode():
                                downloaded = await asyncio.wait_for(
                                    service.download(url, extra_opts=format_opts),
                                    timeout=DOWNLOAD_TIMEOUT_SECONDS,
                                )
                        else:
                            downloaded = await asyncio.wait_for(
                                service.download(url, extra_opts=format_opts),
                                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                            )
                else:
                    downloaded = await asyncio.wait_for(
                        service.download(url, extra_opts=format_opts),
                        timeout=DOWNLOAD_TIMEOUT_SECONDS,
                    )

                async with downloaded:
                    if target_msg:
                        sent = await bot.send_downloaded_media(
                            target_msg,
                            downloaded,
                            is_audio_only=(quality == "audio"),
                            cache_key=cache_key,
                        )
                        if sent:
                            await target_msg.delete()
        except TimeoutError:
            logger.warning("Download timed out for YouTube %s (%s)", quality, video_id)
            if target_msg:
                await set_message_reaction(target_msg, "❌")
                await target_msg.edit_text(
                    f"⏱ Download timed out for {label} (exceeded 5 minutes).\n"
                    "The video stream may be too large or the server connection is slow."
                )
        except Exception as e:
            logger.exception("Failed to download YouTube %s for %s", quality, video_id)
            if target_msg:
                await set_message_reaction(target_msg, "❌")
                await target_msg.edit_text(f"Failed to download YouTube media: {e}")
