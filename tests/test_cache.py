from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Chat, Message, User, Video

from downloader.models import DownloadedMedia
from services.cache import (
    CachedMedia,
    MediaCache,
    get_canonical_cache_key,
    normalize_media_url,
)


def create_user_message(text: str = "") -> Message:
    return Message(
        message_id=1,
        date=1700000000,
        chat=Chat(id=123456, type="private"),
        from_user=User(id=123456, is_bot=False, first_name="Tester"),
        text=text,
    )


def test_url_normalization_instagram():
    # Reels with tracking params
    u1 = "https://www.instagram.com/reel/Dcl5gYWISef/?rpxt=MWNlaWRmdzNhamc2dw=="
    u2 = "https://instagram.com/reel/Dcl5gYWISef?igsh=MXNmZ...&utm_source=copy"
    u3 = "https://www.instagram.com/p/Dcl5gYWISef/?utm_medium=share_sheet"
    u4 = "https://www.instagram.com/share/reel/Dcl5gYWISef/"

    assert normalize_media_url(u1) == "instagram:Dcl5gYWISef"
    assert normalize_media_url(u2) == "instagram:Dcl5gYWISef"
    assert normalize_media_url(u3) == "instagram:Dcl5gYWISef"
    assert normalize_media_url(u4) == "instagram:Dcl5gYWISef"


def test_url_normalization_youtube_shorts():
    u1 = "https://www.youtube.com/shorts/3i_p5Yl1tL0?feature=share&si=abc123"
    u2 = "https://youtube.com/shorts/3i_p5Yl1tL0"
    u3 = "https://m.youtube.com/shorts/3i_p5Yl1tL0/"

    assert normalize_media_url(u1) == "youtube:short:3i_p5Yl1tL0"
    assert normalize_media_url(u2) == "youtube:short:3i_p5Yl1tL0"
    assert normalize_media_url(u3) == "youtube:short:3i_p5Yl1tL0"


def test_url_normalization_youtube_videos():
    u1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&si=12345&feature=shared"
    u2 = "https://youtu.be/dQw4w9WgXcQ?si=abcdef"

    assert normalize_media_url(u1) == "youtube:dQw4w9WgXcQ"
    assert normalize_media_url(u2) == "youtube:dQw4w9WgXcQ"
    assert normalize_media_url(u1, quality="1080") == "youtube:dQw4w9WgXcQ:1080"
    assert normalize_media_url(u2, quality="audio") == "youtube:dQw4w9WgXcQ:audio"


def test_url_normalization_tiktok():
    u1 = "https://www.tiktok.com/@creator/video/7429182371928371928?is_from_webapp=1&sender_device=pc"
    u2 = "https://www.tiktok.com/@creator/photo/7429182371928371928"
    u3 = "https://vt.tiktok.com/ZSxxxxxx/?utm_campaign=share"

    assert normalize_media_url(u1) == "tiktok:video:7429182371928371928"
    assert normalize_media_url(u2) == "tiktok:photo:7429182371928371928"
    assert normalize_media_url(u3) == "tiktok:shortlink:ZSxxxxxx"


def test_url_normalization_generic():
    u = "https://example.com/media/file.mp4?utm_source=twitter&utm_medium=cpc&custom_id=99"
    normalized = normalize_media_url(u)
    assert "utm_source" not in normalized
    assert "utm_medium" not in normalized
    assert "custom_id=99" in normalized


@pytest.mark.asyncio
async def test_get_canonical_cache_key_tiktok_resolution():
    shortlink = "https://vt.tiktok.com/ZSabcdef/"
    resolved = "https://www.tiktok.com/@user/video/7412345678901234567?_t=8"

    with patch("services.cache.resolve_tiktok_url", return_value=resolved):
        key = await get_canonical_cache_key(shortlink, platform="tiktok")
        assert key == "tiktok:video:7412345678901234567"


@pytest.mark.asyncio
async def test_media_cache_crud():
    cache = MediaCache(db_path=":memory:")

    # Initially empty
    assert await cache.count() == 0
    assert await cache.get("nonexistent") is None

    # Store a video
    await cache.set(
        cache_key="test:video:1",
        media_type="video",
        file_id="BAACAgIAAxkBAAI...",
        caption="Epic Reel",
        extra_data={"duration": 15, "width": 1080, "height": 1920},
    )

    assert await cache.count() == 1
    cached = await cache.get("test:video:1")
    assert cached is not None
    assert cached.cache_key == "test:video:1"
    assert cached.media_type == "video"
    assert cached.file_id == "BAACAgIAAxkBAAI..."
    assert cached.caption == "Epic Reel"
    assert cached.extra_data["duration"] == 15

    # Store album
    await cache.set(
        cache_key="test:album:1",
        media_type="album",
        file_id=None,
        caption="Photo Carousel",
        extra_data={"items": [{"type": "photo", "file_id": "AQAD..."}]},
    )
    assert await cache.count() == 2

    # Delete
    deleted = await cache.delete("test:video:1")
    assert deleted is True
    assert await cache.get("test:video:1") is None
    assert await cache.count() == 1


@pytest.mark.asyncio
async def test_media_cache_expired_cleanup():
    cache = MediaCache(db_path=":memory:")

    # Insert an entry with an old timestamp
    await cache.set("old:entry", "video", "file123")
    with patch("time.time", return_value=1000000000.0):
        await cache.set("new:entry", "video", "file456")

    # Clear entries older than 1 day
    with patch("time.time", return_value=1000000000.0 + 86400 * 2):
        purged = await cache.clear_expired(max_age_days=1)
        assert purged >= 1


@pytest.mark.asyncio
async def test_send_cached_media_video():
    from bot import send_cached_media

    cached = CachedMedia(
        cache_key="video:123",
        media_type="video",
        file_id="tg_video_file_id_xyz",
        caption="Test Video",
        extra_data={"duration": 12, "width": 1080, "height": 1920},
        created_at=100.0,
    )

    msg = create_user_message()
    with patch.object(Message, "reply_video", new_callable=AsyncMock) as mock_reply:
        res = await send_cached_media(msg, cached)
        assert res is True
        mock_reply.assert_awaited_once_with(
            video="tg_video_file_id_xyz",
            caption="Test Video",
            duration=12,
            width=1080,
            height=1920,
            supports_streaming=True,
        )


@pytest.mark.asyncio
async def test_send_cached_media_album():
    from bot import send_cached_media

    cached = CachedMedia(
        cache_key="album:456",
        media_type="album",
        file_id=None,
        caption="Test Carousel",
        extra_data={
            "items": [
                {"type": "photo", "file_id": "photo_id_1"},
                {"type": "video", "file_id": "video_id_2"},
            ],
            "audio_file_id": "audio_track_id_3",
            "audio_title": "Soundtrack",
            "audio_performer": "Artist",
        },
        created_at=100.0,
    )

    msg = create_user_message()
    with (
        patch.object(Message, "reply_media_group", new_callable=AsyncMock) as mock_album,
        patch.object(Message, "reply_audio", new_callable=AsyncMock) as mock_audio,
    ):
        res = await send_cached_media(msg, cached)
        assert res is True
        mock_album.assert_awaited_once()
        mock_audio.assert_awaited_once_with(
            audio="audio_track_id_3",
            title="Soundtrack",
            performer="Artist",
        )


@pytest.mark.asyncio
async def test_send_downloaded_media_saves_to_cache(tmp_path: Path):
    from bot import media_cache, send_downloaded_media

    # Create dummy video file
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"dummy mp4 content")

    downloaded = DownloadedMedia(
        filepath=video_file,
        duration=10.0,
        width=1280.0,
        height=720.0,
        title="Saved Video",
    )

    msg = create_user_message()
    fake_sent_msg = Message(
        message_id=2,
        date=1700000000,
        chat=Chat(id=123456, type="private"),
        video=Video(
            file_id="new_telegram_file_id",
            file_unique_id="unique_123",
            width=1280,
            height=720,
            duration=10,
        ),
    )

    with (
        patch.object(media_cache, "set", new_callable=AsyncMock) as mock_cache_set,
        patch.object(Message, "reply_video", new_callable=AsyncMock, return_value=fake_sent_msg),
    ):
        await send_downloaded_media(msg, downloaded, cache_key="youtube:short:abc12345")
        mock_cache_set.assert_awaited_once()
        _, kwargs = mock_cache_set.call_args
        assert kwargs["cache_key"] == "youtube:short:abc12345"
        assert kwargs["file_id"] == "new_telegram_file_id"
        assert kwargs["media_type"] == "video"


@pytest.mark.asyncio
async def test_media_url_handler_cache_hit_bypasses_download():
    from bot import media_cache, media_url_handler, platform_registry

    await media_cache.set(
        cache_key="instagram:Dcl5gYWISef",
        media_type="video",
        file_id="cached_ig_file_id",
        caption="Cached Reel",
    )

    msg = create_user_message("https://www.instagram.com/reel/Dcl5gYWISef/?rpxt=123")
    service = platform_registry.get_service_by_name("instagram")

    with (
        patch(
            "bot.send_cached_media", new_callable=AsyncMock, return_value=True
        ) as mock_send_cached,
        patch.object(service, "download", new_callable=AsyncMock) as mock_download,
    ):
        await media_url_handler(
            msg, "https://www.instagram.com/reel/Dcl5gYWISef/?rpxt=123", "instagram"
        )
        mock_send_cached.assert_awaited_once()
        mock_download.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_url_handler_cache_fallback_on_telegram_error(tmp_path: Path):
    from bot import media_cache, media_url_handler, platform_registry

    await media_cache.set(
        cache_key="instagram:Fallback123",
        media_type="video",
        file_id="bad_file_id",
        caption="Bad Reel",
    )

    msg = create_user_message("https://www.instagram.com/reel/Fallback123/")
    service = platform_registry.get_service_by_name("instagram")

    dummy_file = tmp_path / "reel.mp4"
    dummy_file.write_bytes(b"content")
    downloaded = DownloadedMedia(filepath=dummy_file, duration=5.0, title="Reel")

    from aiogram.methods import SendVideo

    with (
        patch(
            "bot.send_cached_media",
            side_effect=TelegramBadRequest(
                method=SendVideo(chat_id=1, video="bad"), message="Wrong file_id"
            ),
        ),
        patch.object(
            service, "download", new_callable=AsyncMock, return_value=downloaded
        ) as mock_download,
        patch("bot.send_downloaded_media", new_callable=AsyncMock, return_value=True),
        patch.object(Message, "reply", new_callable=AsyncMock) as mock_reply,
    ):
        mock_status = AsyncMock()
        mock_reply.return_value = mock_status

        await media_url_handler(msg, "https://www.instagram.com/reel/Fallback123/", "instagram")

        # Verified fallback occurred:
        mock_download.assert_awaited_once()
        # Invalid entry should be purged from cache
        assert await media_cache.get("instagram:Fallback123") is None


@pytest.mark.asyncio
async def test_youtube_callback_cache_hit():
    from aiogram.types import CallbackQuery

    from bot import media_cache, youtube_callback_handler

    await media_cache.set(
        cache_key="youtube:abc12345:1080",
        media_type="video",
        file_id="cached_yt_1080_file_id",
        caption="Cached 1080p Video",
    )

    target_msg = create_user_message()
    cb = CallbackQuery(
        id="cb_test_yt",
        from_user=User(id=123456, is_bot=False, first_name="Tester"),
        chat_instance="inst",
        message=target_msg,
        data="yt:1080:abc12345",
    )

    with (
        patch(
            "bot.send_cached_media", new_callable=AsyncMock, return_value=True
        ) as mock_send_cached,
        patch.object(Message, "delete", new_callable=AsyncMock) as mock_delete,
    ):
        await youtube_callback_handler(cb)
        mock_send_cached.assert_awaited_once()
        mock_delete.assert_awaited_once()
