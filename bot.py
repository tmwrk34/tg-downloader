import asyncio
import logging
import time
import uuid

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InputMediaPhoto,
    InputMediaVideo,
    InputRichBlockPhoto,
    InputRichBlockSlideshow,
    InputRichBlockUnion,
    InputRichBlockVideo,
    InputRichMessage,
    InputTextMessageContent,
    Message,
    RichBlockCaption,
)

from config import load_config
from downloader import (
    DownloadedMedia,
    MediaDownloaderCore,
    MediaMetadata,
    PlatformServiceRegistry,
    YouTubeService,
)
from filters import DiceFilter, MediaUrlFilter, TriggerPhraseFilter
from middlewares import WhitelistMiddleware
from services.cache import (
    CachedMedia,
    MediaCache,
    get_canonical_cache_key,
)
from services.dice import DiceParser, get_secret_roll, store_secret_roll

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
media_cache = MediaCache()

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


@dp.message(CommandStart())
async def cmd_start(message: Message, is_admin: bool = False, is_private: bool = False):
    first_name = message.from_user.first_name if message.from_user else "user"
    status_suffix = " 👑 (Admin/Dev)" if is_admin and is_private else ""

    await message.answer(
        f"Hello, {first_name}! 👋{status_suffix}\n\n"
        "🎬 <b>Media Downloader:</b> Send any supported TikTok, Instagram, or YouTube link.\n"
        "🎲 <b>Dice Roller:</b> Use <code>/roll [expression]</code> (e.g. <code>2d6+3</code>, <code>4d6k3</code>) or type dice in chat!\n"
        "🔒 <b>Secret Rolls:</b> Use <code>/dmroll [expression]</code> for hidden rolls.\n"
        "✨ <b>Inline Mode:</b> Type <code>@bot</code> in any chat to roll dice.\n"
        "📖 <b>Help:</b> Type <code>/help</code> for full documentation and options.",
        parse_mode="HTML",
    )


HELP_MANUAL_HTML = (
    "📖 <b>Bard-Bot Commands & Features Manual</b>\n\n"
    "🎬 <b>Media Downloader:</b>\n"
    "Send any supported link directly into the chat:\n"
    "• <b>TikTok:</b> Videos, short links (<code>vt.tiktok.com</code>), and Photo slideshows (with audio).\n"
    "• <b>Instagram:</b> Reels, Posts, and Carousel albums (<code>/reel/</code>, <code>/p/</code>).\n"
    "• <b>YouTube Shorts:</b> Instant direct video download.\n"
    "• <b>YouTube Videos:</b> Quality selector (1080p, 720p, 360p) or MP3 audio extraction with album art.\n"
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
    "🏓 <b>Utilities:</b>\n"
    "• <code>/help</code> — Show this manual.\n"
    "• <code>ping</code> — Check bot connection and status."
)


@dp.message(Command("help", "хелп", "довідка", "помощь"))
@dp.message(TriggerPhraseFilter(category="help", exact_match=True))
async def help_command_handler(message: Message):
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
    await message.reply(HELP_MANUAL_HTML, reply_markup=keyboard, parse_mode="HTML")


@dp.message(TriggerPhraseFilter(category="ping", exact_match=True))
async def ping_handler(message: Message, matched_phrase: str = ""):
    await message.reply("Pong! 🏓 Alive, well, and comfortably running inside a Raspberry Pi.")


@dp.message(Command("roll", "ролл", "кинь", "кидок"))
async def roll_command_handler(message: Message, command: CommandObject):
    expr = command.args.strip() if command.args else "d20"
    user_name = message.from_user.first_name if message.from_user else "Player"
    try:
        res = DiceParser.parse_and_roll(expr)
        await message.reply(res.format_html(user_name), parse_mode="HTML")
    except ValueError as err:
        await message.reply(f"⚠️ <b>Invalid roll syntax:</b> {err}", parse_mode="HTML")


@dp.message(Command("dmroll", "дмролл", "дм", "секретролл"))
async def dmroll_command_handler(message: Message, command: CommandObject):
    expr = command.args.strip() if command.args else "d20"
    user_name = message.from_user.first_name if message.from_user else "Player"
    user_id = message.from_user.id if message.from_user else 0

    try:
        res = DiceParser.parse_and_roll(expr)
    except ValueError as err:
        await message.reply(f"⚠️ <b>Invalid roll syntax:</b> {err}", parse_mode="HTML")
        return

    if message.chat.type == "private":
        await message.reply(res.format_html(user_name), parse_mode="HTML")
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
    await message.reply(
        f"🎲 <b>Secret Roll made by {user_name}</b>\n<i>Only {user_name} can view this result.</i>",
        reply_markup=keyboard,
        parse_mode="HTML",
    )


@dp.message(DiceFilter())
async def natural_dice_handler(message: Message, dice_expr: str, is_secret: bool = False):
    user_name = message.from_user.first_name if message.from_user else "Player"
    user_id = message.from_user.id if message.from_user else 0

    try:
        res = DiceParser.parse_and_roll(dice_expr)
    except ValueError:
        return

    if is_secret:
        if message.chat.type == "private":
            await message.reply(res.format_html(user_name), parse_mode="HTML")
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
        await message.reply(
            f"🎲 <b>Secret Roll made by {user_name}</b>\n<i>Only {user_name} can view this result.</i>",
            reply_markup=keyboard,
            parse_mode="HTML",
        )
    else:
        await message.reply(res.format_html(user_name), parse_mode="HTML")


@dp.callback_query(F.data.startswith("secret_roll:"))
async def secret_roll_callback_handler(callback_query: CallbackQuery):
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


@dp.inline_query()
async def inline_dice_handler(inline_query: InlineQuery):
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
    else:
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

    await inline_query.answer(results=results, cache_time=1, is_personal=True)


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
                await media_cache.set(
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
                await media_cache.set(
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
                await media_cache.set(
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
                await media_cache.set(
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


@dp.message(MediaUrlFilter())
async def media_url_handler(message: Message, url: str, platform: str):
    logger.info("Matched %s URL: %s", platform, url)

    service = platform_registry.get_service_by_name(
        platform
    ) or platform_registry.get_service_for_url(url)
    if not service:
        await message.reply("Unsupported platform.")
        return

    # Check cache for immediate delivery (TikTok, Instagram, YouTube Shorts, etc.)
    # (except for long-form YouTube videos where quality selection is prompted first)
    is_yt_long = (
        platform == "youtube"
        and isinstance(service, YouTubeService)
        and not YouTubeService.is_short(url)
    )

    cache_key: str | None = None
    if not is_yt_long:
        cache_key = await get_canonical_cache_key(url, platform=platform)
        cached = await media_cache.get(cache_key)
        if cached:
            try:
                sent = await send_cached_media(message, cached)
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
                await media_cache.delete(cache_key)

    # For YouTube: prompt user with resolution and audio options via inline keyboard
    # (except for YouTube Shorts which are downloaded immediately like TikTok/Instagram)
    if is_yt_long:
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

    # For TikTok, Instagram, YouTube Shorts, and direct platforms: download immediately
    label = (
        "YouTube Short"
        if (platform == "youtube" and YouTubeService.is_short(url))
        else platform.capitalize()
    )
    status_msg = await message.reply(f"Downloading {label} media... Please wait ⏳")
    try:
        downloaded = await asyncio.wait_for(
            service.download(url),
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        )
        async with downloaded:
            sent = await send_downloaded_media(status_msg, downloaded, cache_key=cache_key)
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
    cache_key = f"youtube:{video_id}:{quality}"
    cached = await media_cache.get(cache_key)
    if cached:
        try:
            if callback.message and isinstance(callback.message, Message):
                sent = await send_cached_media(callback.message, cached)
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
            await media_cache.delete(cache_key)

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
                    callback.message,
                    downloaded,
                    is_audio_only=(quality == "audio"),
                    cache_key=cache_key,
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
