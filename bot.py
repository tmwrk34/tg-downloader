import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile, Message

from config import load_config
from downloader import MediaDownloaderCore, PlatformServiceRegistry
from filters import MediaUrlFilter, TriggerPhraseFilter
from middlewares import WhitelistMiddleware

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
    status_msg = await message.reply(f"Downloading {platform.capitalize()} media... Please wait ⏳")

    service = platform_registry.get_service_by_name(
        platform
    ) or platform_registry.get_service_for_url(url)
    if not service:
        await status_msg.edit_text("Unsupported platform.")
        return

    try:
        downloaded = await service.download(url)
        async with downloaded:
            caption = f"🎬 {downloaded.title}" if downloaded.title else ""
            file_input = FSInputFile(downloaded.filepath)
            ext = downloaded.filepath.suffix.lower()

            if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                await message.reply_photo(photo=file_input, caption=caption)
            elif ext in {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac"}:
                await message.reply_audio(audio=file_input, caption=caption)
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
    except Exception as e:
        logger.exception("Failed to process media URL %s", url)
        await status_msg.edit_text(f"Failed to download media: {e}")


async def main():
    bot = Bot(token=config.bot_token)
    logger.info("Bot successfully started. Press Ctrl+C to stop.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
