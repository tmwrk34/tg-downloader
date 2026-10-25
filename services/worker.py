"""
Bounded asyncio semaphore, singleflight request deduplication, and task queue
management for media downloading and FFmpeg transcoding.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

from aiogram.enums import ChatAction
from aiogram.types import Message, ReactionTypeEmoji

logger = logging.getLogger(__name__)


async def set_message_reaction(message: Message, emoji: str | None) -> bool:
    """
    Safely set or clear a native Telegram reaction on a message.
    Pass `None` to remove the bot's reaction.
    Never raises exceptions if reactions are disabled or unsupported in the chat.
    """
    try:
        reactions = [ReactionTypeEmoji(emoji=emoji)] if emoji else []
        await message.react(reaction=reactions)
        return True
    except Exception as e:
        logger.debug("Could not set reaction '%s' on message: %s", emoji, e)
        return False


@asynccontextmanager
async def chat_action_heartbeat(
    message: Message,
    action: ChatAction | str = ChatAction.UPLOAD_VIDEO,
    interval: float = 4.0,
) -> AsyncIterator[None]:
    """
    Continuously emit a native Telegram chat action (e.g. upload_video, record_voice)
    in the background while a long-running media task is actively processing.
    """

    async def _loop() -> None:
        while True:
            try:
                bot = getattr(message, "bot", None)
                if bot is not None:
                    await bot.send_chat_action(chat_id=message.chat.id, action=action)
                else:
                    await message.answer_chat_action(action=action)
            except Exception as e:
                logger.debug("Chat action heartbeat skipped: %s", e)
            await asyncio.sleep(interval)

    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


class MediaWorkerLimiter:
    """
    Concurrency controller managing bounded semaphores for network downloads
    and CPU-intensive FFmpeg transcodes, plus per-URL singleflight locks.
    """

    def __init__(
        self,
        max_concurrent_downloads: int = 2,
        max_concurrent_transcodes: int = 1,
    ) -> None:
        if max_concurrent_downloads < 1:
            raise ValueError("max_concurrent_downloads must be >= 1")
        if max_concurrent_transcodes < 1:
            raise ValueError("max_concurrent_transcodes must be >= 1")

        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_transcodes = max_concurrent_transcodes

        self._download_sem = asyncio.Semaphore(max_concurrent_downloads)
        self._transcode_sem = asyncio.Semaphore(max_concurrent_transcodes)

        self._active_downloads = 0
        self._queued_downloads = 0
        self._active_transcodes = 0
        self._queued_transcodes = 0

        # Singleflight locks per canonical cache_key: key -> (Lock, ref_count)
        self._inflight_locks: dict[str, tuple[asyncio.Lock, int]] = {}

    @property
    def active_downloads(self) -> int:
        return self._active_downloads

    @property
    def queued_downloads(self) -> int:
        return self._queued_downloads

    @property
    def active_transcodes(self) -> int:
        return self._active_transcodes

    @property
    def queued_transcodes(self) -> int:
        return self._queued_transcodes

    @property
    def is_download_busy(self) -> bool:
        """Return True if all download worker slots are currently occupied."""
        return self._download_sem.locked()

    @property
    def is_transcode_busy(self) -> bool:
        """Return True if all transcode worker slots are currently occupied."""
        return self._transcode_sem.locked()

    @asynccontextmanager
    async def singleflight(self, key: str | None) -> AsyncIterator[None]:
        """
        Deduplicate concurrent requests for the same canonical cache key so
        identical links posted simultaneously do not trigger duplicate downloads.
        """
        if not key:
            yield
            return

        entry = self._inflight_locks.get(key)
        if entry is None:
            lock = asyncio.Lock()
            self._inflight_locks[key] = (lock, 1)
        else:
            lock, count = entry
            self._inflight_locks[key] = (lock, count + 1)

        try:
            async with lock:
                yield
        finally:
            current = self._inflight_locks.get(key)
            if current is not None:
                lck, cnt = current
                if cnt <= 1:
                    self._inflight_locks.pop(key, None)
                else:
                    self._inflight_locks[key] = (lck, cnt - 1)

    @asynccontextmanager
    async def acquire_download(
        self,
        on_queued: Callable[[int], Awaitable[Any]] | None = None,
        on_active: Callable[[], Awaitable[Any]] | None = None,
    ) -> AsyncIterator[int]:
        """
        Acquire a download worker slot.
        If all slots are busy, invokes `on_queued(queue_position)` before waiting.
        Once acquired, invokes `on_active()` and yields the 1-based active worker count.
        """
        was_queued = False
        if self._download_sem.locked():
            self._queued_downloads += 1
            was_queued = True
            queue_pos = self._queued_downloads
            logger.info(
                "Download queue busy (%d/%d active). Task queued at position #%d.",
                self._active_downloads,
                self.max_concurrent_downloads,
                queue_pos,
            )
            if on_queued is not None:
                with suppress(Exception):
                    await on_queued(queue_pos)

        try:
            await self._download_sem.acquire()
        except BaseException:
            if was_queued:
                self._queued_downloads = max(0, self._queued_downloads - 1)
            raise

        if was_queued:
            self._queued_downloads = max(0, self._queued_downloads - 1)
        self._active_downloads += 1

        try:
            if on_active is not None:
                with suppress(Exception):
                    await on_active()
            yield self._active_downloads
        finally:
            self._active_downloads = max(0, self._active_downloads - 1)
            self._download_sem.release()

    @asynccontextmanager
    async def acquire_transcode(
        self,
        on_queued: Callable[[int], Awaitable[Any]] | None = None,
        on_active: Callable[[], Awaitable[Any]] | None = None,
    ) -> AsyncIterator[int]:
        """
        Acquire a CPU transcode worker slot for heavy FFmpeg processing.
        """
        was_queued = False
        if self._transcode_sem.locked():
            self._queued_transcodes += 1
            was_queued = True
            queue_pos = self._queued_transcodes
            logger.info(
                "Transcode queue busy (%d/%d active). Task queued at position #%d.",
                self._active_transcodes,
                self.max_concurrent_transcodes,
                queue_pos,
            )
            if on_queued is not None:
                with suppress(Exception):
                    await on_queued(queue_pos)

        try:
            await self._transcode_sem.acquire()
        except BaseException:
            if was_queued:
                self._queued_transcodes = max(0, self._queued_transcodes - 1)
            raise

        if was_queued:
            self._queued_transcodes = max(0, self._queued_transcodes - 1)
        self._active_transcodes += 1

        try:
            if on_active is not None:
                with suppress(Exception):
                    await on_active()
            yield self._active_transcodes
        finally:
            self._active_transcodes = max(0, self._active_transcodes - 1)
            self._transcode_sem.release()
