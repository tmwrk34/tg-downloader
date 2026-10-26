from __future__ import annotations

import logging
import sys
from pathlib import Path

import aiohttp
from aiogram import Bot
from aiogram.types import PhotoSize

logger = logging.getLogger(__name__)


async def download_telegram_photo(active_bot: Bot, photo: PhotoSize) -> bytes | None:
    """Download photo bytes safely, supporting direct local filesystem access and HTTP fallback."""
    try:
        file_info = await active_bot.get_file(photo.file_id)
        if not file_info.file_path:
            return None

        # 1. Direct local file check (e.g. shared volume with local telegram-bot-api)
        try:
            local_candidate = Path(file_info.file_path)
            if local_candidate.is_file():
                return local_candidate.read_bytes()
        except (PermissionError, OSError) as perm_err:
            logger.warning("Direct local photo read failed due to permissions/OS: %s", perm_err)

        # 2. Try standard aiogram download
        try:
            file_io = await active_bot.download(file_info)
            if file_io:
                return file_io.getvalue() if hasattr(file_io, "getvalue") else file_io.read()
        except Exception as dl_err:
            logger.debug("Standard bot.download failed: %s", dl_err)

        # 3. If running with local_bot_api, extract relative path from absolute path
        bot_mod = sys.modules.get("bot")
        cfg = getattr(bot_mod, "config", None)
        local_url = getattr(cfg, "local_bot_api_url", None) if cfg else None
        if local_url:
            rel_path = file_info.file_path.lstrip("/")
            for marker in ("photos/", "documents/", "video/", "voice/", "video_notes/"):
                if marker in rel_path:
                    rel_path = rel_path[rel_path.index(marker) :]
                    break
            download_url = f"{local_url.rstrip('/')}/file/bot{active_bot.token}/{rel_path}"
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    logger.warning(
                        "Local bot API photo fetch returned %s for %s", resp.status, download_url
                    )
    except Exception as e:
        logger.warning("Failed to download photo for Munito AI: %s", e)
    return None
