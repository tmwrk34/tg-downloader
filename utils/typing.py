from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiogram import Bot


@asynccontextmanager
async def keep_typing(
    bot: Bot,
    chat_id: int,
    action: str = "typing",
    interval: float = 4.0,
) -> AsyncIterator[None]:
    """
    Asynchronously keep sending a chat action (e.g. 'typing', 'upload_video')
    at regular intervals until the context block exits.
    """
    stop_event = asyncio.Event()

    async def _worker() -> None:
        while not stop_event.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=action)
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except TimeoutError:
                pass

    task = asyncio.create_task(_worker())
    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
