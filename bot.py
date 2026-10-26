from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer

from config import load_config
from downloader import (
    MediaDownloaderCore,
    PlatformServiceRegistry,
)
from middlewares import WhitelistMiddleware
from services.cache import MediaCache
from services.chatter import ChatterService
from services.munito import MunitoAIService
from services.worker import MediaWorkerLimiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose event logs for every handled/unhandled update
logging.getLogger("aiogram.event").setLevel(logging.WARNING)

# Load configuration
config = load_config()

# Initialize core services
downloader_core = MediaDownloaderCore(download_dir="./downloads")
platform_registry = PlatformServiceRegistry(downloader_core)
media_cache = MediaCache()
chatter_service = ChatterService()
munito_service = MunitoAIService(
    api_key=config.munito_api_key,
    model=config.munito_model,
    base_url=config.munito_base_url,
    system_prompt=config.munito_system_prompt,
)
worker_limiter = MediaWorkerLimiter(
    max_concurrent_downloads=config.max_concurrent_downloads,
    max_concurrent_transcodes=config.max_concurrent_transcodes,
)


async def _get_cached_bot_user(bot: Bot) -> Any:
    """Fetch and cache bot user information on the Bot instance."""
    inst_dict = getattr(bot, "__dict__", None)
    if isinstance(inst_dict, dict) and "_cached_user" in inst_dict:
        return inst_dict["_cached_user"]

    try:
        user = await bot.get_me()
        if isinstance(inst_dict, dict):
            inst_dict["_cached_user"] = user
        return user
    except Exception as e:
        logger.debug("Could not fetch bot user: %s", e)
        return None


# Import domain handlers and filters after core services are instantiated
from filters.admin import is_chat_admin as _is_user_chatter_admin  # noqa: E402
from handlers import (  # noqa: E402
    ai_router,
    base_router,
    chatter_router,
    dice_router,
    downloader_router,
    main_router,
)
from handlers.ai import (  # noqa: E402
    _extract_bot_tag_prompt,
    _handle_ai_mention,
    ai_callback_handler,
    ai_command_handler,
    chatter_message_listener,
)
from handlers.base import (  # noqa: E402
    HELP_MANUAL_HTML,
    cmd_start,
    help_command_handler,
    ping_handler,
)
from handlers.chatter import (  # noqa: E402
    COOLDOWN_STEPS,
    REACT_CHANCE_STEPS,
    REPLY_CHANCE_STEPS,
    _next_step,
    build_chatter_keyboard,
    chatter_callback_handler,
    chatter_command_handler,
    format_chatter_card,
)
from handlers.dice import (  # noqa: E402
    dmroll_command_handler,
    inline_dice_handler,
    natural_dice_handler,
    roll_command_handler,
    secret_roll_callback_handler,
)
from handlers.downloader import (  # noqa: E402
    format_duration,
    media_url_handler,
    send_cached_media,
    send_downloaded_media,
    youtube_callback_handler,
)
from utils.telegram_media import download_telegram_photo as _download_telegram_photo  # noqa: E402

# Initialize Dispatcher and register middlewares
dp = Dispatcher()

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

# Register main router containing all domain handlers
dp.include_router(main_router)


async def main() -> None:
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

__all__ = [
    "COOLDOWN_STEPS",
    "HELP_MANUAL_HTML",
    "REACT_CHANCE_STEPS",
    "REPLY_CHANCE_STEPS",
    "_download_telegram_photo",
    "_extract_bot_tag_prompt",
    "_get_cached_bot_user",
    "_handle_ai_mention",
    "_is_user_chatter_admin",
    "_next_step",
    "ai_callback_handler",
    "ai_command_handler",
    "ai_router",
    "base_router",
    "build_chatter_keyboard",
    "chatter_callback_handler",
    "chatter_command_handler",
    "chatter_message_listener",
    "chatter_router",
    "chatter_service",
    "cmd_start",
    "config",
    "dice_router",
    "dmroll_command_handler",
    "downloader_core",
    "downloader_router",
    "dp",
    "format_chatter_card",
    "format_duration",
    "help_command_handler",
    "inline_dice_handler",
    "main",
    "media_cache",
    "media_url_handler",
    "munito_service",
    "natural_dice_handler",
    "ping_handler",
    "platform_registry",
    "roll_command_handler",
    "secret_roll_callback_handler",
    "send_cached_media",
    "send_downloaded_media",
    "worker_limiter",
    "youtube_callback_handler",
]
