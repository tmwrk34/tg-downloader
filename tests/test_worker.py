import asyncio
import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from aiogram.enums import ChatAction
from aiogram.types import Chat, Message, User

from downloader.core import MediaDownloaderCore
from downloader.platforms.tiktok import _shortlink_cache, resolve_tiktok_url
from services.cache import MediaCache
from services.worker import (
    MediaWorkerLimiter,
    chat_action_heartbeat,
    set_message_reaction,
)


def _make_message(chat_id: int = 12345, message_id: int = 99) -> Message:
    bot = MagicMock(spec=Bot)
    bot.send_chat_action = AsyncMock()
    msg = Message(
        message_id=message_id,
        date=datetime.datetime.now(datetime.UTC),
        chat=Chat(id=chat_id, type="private"),
        from_user=User(id=chat_id, is_bot=False, first_name="Tester"),
        text="https://www.tiktok.com/@user/video/111",
    )
    return msg.as_(bot)


@pytest.mark.asyncio
async def test_worker_limiter_bounded_downloads():
    limiter = MediaWorkerLimiter(max_concurrent_downloads=2, max_concurrent_transcodes=1)
    assert limiter.max_concurrent_downloads == 2
    assert limiter.max_concurrent_transcodes == 1

    queued_positions: list[int] = []
    started_order: list[int] = []
    release_event = asyncio.Event()

    async def worker_task(task_id: int):
        async def on_queued(pos: int):
            queued_positions.append(pos)

        async with limiter.acquire_download(on_queued=on_queued):
            started_order.append(task_id)
            await release_event.wait()

    # Start 2 tasks that immediately occupy both download slots
    t1 = asyncio.create_task(worker_task(1))
    t2 = asyncio.create_task(worker_task(2))
    await asyncio.sleep(0.02)

    assert limiter.active_downloads == 2
    assert limiter.queued_downloads == 0
    assert started_order == [1, 2]
    assert queued_positions == []

    # Start a 3rd task — must queue and report position 1
    t3 = asyncio.create_task(worker_task(3))
    await asyncio.sleep(0.02)

    assert limiter.active_downloads == 2
    assert limiter.queued_downloads == 1
    assert queued_positions == [1]
    assert started_order == [1, 2]

    # Release slots and ensure t3 completes
    release_event.set()
    await asyncio.gather(t1, t2, t3)

    assert limiter.active_downloads == 0
    assert limiter.queued_downloads == 0
    assert started_order == [1, 2, 3]


@pytest.mark.asyncio
async def test_worker_limiter_bounded_transcodes():
    limiter = MediaWorkerLimiter(max_concurrent_downloads=2, max_concurrent_transcodes=1)
    release_event = asyncio.Event()
    active_snapshots: list[int] = []

    async def transcode_job():
        async with limiter.acquire_transcode():
            active_snapshots.append(limiter.active_transcodes)
            await release_event.wait()

    t1 = asyncio.create_task(transcode_job())
    t2 = asyncio.create_task(transcode_job())
    await asyncio.sleep(0.02)

    assert limiter.active_transcodes == 1
    assert limiter.queued_transcodes == 1

    release_event.set()
    await asyncio.gather(t1, t2)

    assert active_snapshots == [1, 1]
    assert limiter.active_transcodes == 0
    assert limiter.queued_transcodes == 0


@pytest.mark.asyncio
async def test_worker_limiter_singleflight_deduplication():
    limiter = MediaWorkerLimiter(max_concurrent_downloads=4, max_concurrent_transcodes=2)
    execution_log: list[str] = []

    async def job(name: str, key: str):
        async with limiter.singleflight(key):
            execution_log.append(f"{name}_start")
            await asyncio.sleep(0.03)
            execution_log.append(f"{name}_end")

    # Two tasks with identical key must serialize; a task with different key runs concurrently
    await asyncio.gather(
        job("dup1", "tiktok:https://www.tiktok.com/@u/video/1"),
        job("dup2", "tiktok:https://www.tiktok.com/@u/video/1"),
        job("other", "tiktok:https://www.tiktok.com/@u/video/2"),
    )

    # Verify dup1 completed before dup2 started
    assert execution_log.index("dup1_end") < execution_log.index("dup2_start")
    # Lock dictionary should be cleaned up when all waiters finish
    assert len(limiter._inflight_locks) == 0


@pytest.mark.asyncio
async def test_set_message_reaction_lifecycle():
    msg = _make_message()
    with patch.object(Message, "react", new_callable=AsyncMock) as mock_react:
        await set_message_reaction(msg, "⏳")
        assert mock_react.call_count == 1
        reactions = mock_react.call_args.kwargs["reaction"]
        assert len(reactions) == 1
        assert reactions[0].emoji == "⏳"

        # Clear reaction
        await set_message_reaction(msg, None)
        assert mock_react.call_count == 2
        assert mock_react.call_args.kwargs["reaction"] == []

        # Fault tolerance when Telegram rejects reaction in restricted chat
        mock_react.side_effect = RuntimeError("REACTION_INVALID")
        await set_message_reaction(msg, "❌")  # Should not raise


@pytest.mark.asyncio
async def test_chat_action_heartbeat():
    msg = _make_message()
    async with chat_action_heartbeat(msg, ChatAction.UPLOAD_VIDEO, interval=0.02):
        await asyncio.sleep(0.05)

    assert msg.bot.send_chat_action.call_count >= 1
    msg.bot.send_chat_action.assert_called_with(
        chat_id=msg.chat.id,
        action=ChatAction.UPLOAD_VIDEO,
    )


def test_tiktok_shortlink_lru_cache_avoids_duplicate_network_calls():
    _shortlink_cache.clear()
    mock_resp = MagicMock()
    mock_resp.geturl.return_value = "https://www.tiktok.com/@user/video/987654321"
    mock_resp.__enter__.return_value = mock_resp

    short_url = "https://vt.tiktok.com/ZScacheTest1/"
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        first = resolve_tiktok_url(short_url)
        second = resolve_tiktok_url(short_url)

        assert first == "https://www.tiktok.com/@user/video/987654321"
        assert second == "https://www.tiktok.com/@user/video/987654321"
        assert mock_open.call_count == 1


@pytest.mark.asyncio
async def test_media_cache_l1_memory_hit_and_lru_eviction(tmp_path: Path):
    cache = MediaCache(db_path=tmp_path / "l1_test.db", l1_max_size=2)
    await cache.set("tiktok:video:1", "video", file_id="fid_1")
    await cache.set("tiktok:video:2", "video", file_id="fid_2")
    await cache.set("tiktok:video:3", "video", file_id="fid_3")

    # L1 max size is 2, so key 1 was evicted from L1 RAM but remains in SQLite
    assert len(cache._l1_cache) == 2
    assert "tiktok:video:1" not in cache._l1_cache
    entry1 = await cache.get("tiktok:video:1")
    assert entry1 is not None
    assert entry1.file_id == "fid_1"
    # Fetching key 1 promotes it back into L1 RAM cache
    assert len(cache._l1_cache) == 2
    assert "tiktok:video:1" in cache._l1_cache
    cache.close()


def test_stale_job_directories_cleanup_on_core_init(tmp_path: Path):
    stale_job = tmp_path / "job_deadbeef1234"
    stale_job.mkdir()
    (stale_job / "partial.mp4").write_bytes(b"1234")
    keep_file = tmp_path / "keep.txt"
    keep_file.write_text("keep", encoding="utf-8")

    MediaDownloaderCore(download_dir=tmp_path)
    assert not stale_job.exists()
    assert keep_file.exists()
