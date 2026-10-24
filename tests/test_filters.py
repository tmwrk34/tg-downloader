from datetime import UTC, datetime

import pytest
from aiogram.types import Chat, Message, User

from filters import (
    MediaUrlFilter,
    Platform,
    TriggerPhraseFilter,
    TriggerRegistry,
    clean_url,
)


def create_message(text: str | None = None, caption: str | None = None) -> Message:
    return Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=-1002366722481, type="supergroup"),
        from_user=User(id=12345, is_bot=False, first_name="TestUser"),
        text=text,
        caption=caption,
    )


def test_clean_url():
    assert clean_url("https://example.com/video") == "https://example.com/video"
    assert clean_url("https://example.com/video.") == "https://example.com/video"
    assert clean_url("https://example.com/video,") == "https://example.com/video"
    assert clean_url("https://example.com/video!") == "https://example.com/video"
    assert clean_url("https://example.com/video)") == "https://example.com/video"
    assert (
        clean_url("https://example.com/video?param=1&foo=bar!")
        == "https://example.com/video?param=1&foo=bar"
    )
    assert clean_url("  https://example.com/video?a=b  ") == "https://example.com/video?a=b"
    assert clean_url("https://example.com/wiki/Foo_(bar)") == "https://example.com/wiki/Foo_(bar)"


def test_trigger_registry():
    registry = TriggerRegistry()
    registry.register("custom", ["foo", "bar", "FOO"])
    assert registry.get("custom") == {"foo", "bar", "FOO"}
    assert registry.get("CUSTOM") == {"foo", "bar", "FOO"}
    assert registry.get("nonexistent") == set()
    assert "custom" in registry.all_categories()
    assert registry.all_phrases() == {"foo", "bar", "FOO"}


@pytest.mark.asyncio
async def test_trigger_phrase_filter_exact_match():
    filter_obj = TriggerPhraseFilter(phrases=["ping", "pong"], exact_match=True)

    # Exact match cases
    result = await filter_obj(create_message(text="ping"))
    assert result == {"matched_phrase": "ping", "trigger": "ping"}

    result = await filter_obj(create_message(text="  PONG  "))
    assert result == {"matched_phrase": "pong", "trigger": "pong"}

    # Substrings or extra words should fail when exact_match=True
    assert await filter_obj(create_message(text="ping me")) is False
    assert await filter_obj(create_message(text="this is ping")) is False
    assert await filter_obj(create_message(text="something else")) is False


@pytest.mark.asyncio
async def test_trigger_phrase_filter_substring_match():
    filter_obj = TriggerPhraseFilter(phrases=["download", "fetch"], exact_match=False)

    result = await filter_obj(create_message(text="Please Download this for me!"))
    assert result == {"matched_phrase": "download", "trigger": "download"}

    result = await filter_obj(create_message(caption="Can you fetch it?"))
    assert result == {"matched_phrase": "fetch", "trigger": "fetch"}

    assert await filter_obj(create_message(text="Random text here")) is False
    assert await filter_obj(create_message(text=None, caption=None)) is False


@pytest.mark.asyncio
async def test_trigger_phrase_filter_with_registry():
    registry = TriggerRegistry({"admin": {"sudo", "restart"}})
    filter_obj = TriggerPhraseFilter(category="admin", registry=registry, exact_match=True)

    result = await filter_obj(create_message(text="SUDO"))
    assert result == {"matched_phrase": "sudo", "trigger": "sudo"}

    assert await filter_obj(create_message(text="help")) is False


@pytest.mark.asyncio
async def test_trigger_phrase_filter_case_sensitive():
    filter_obj = TriggerPhraseFilter(phrases=["Ping"], exact_match=True, case_sensitive=True)

    assert await filter_obj(create_message(text="Ping")) == {
        "matched_phrase": "Ping",
        "trigger": "Ping",
    }
    assert await filter_obj(create_message(text="ping")) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, platform_expected",
    [
        ("https://www.tiktok.com/@user/video/7123456789012345678", "tiktok"),
        ("https://tiktok.com/@user/video/7123456789012345678?is_from_webapp=1", "tiktok"),
        ("https://www.tiktok.com/@user/photo/7123456789012345678", "tiktok"),
        ("https://vm.tiktok.com/ZM8vABC12/", "tiktok"),
        ("https://vt.tiktok.com/ZS8vABC12/", "tiktok"),
        ("https://m.tiktok.com/v/7123456789012345678.html", "tiktok"),
        ("https://www.tiktok.com/t/ZT8vABC12/", "tiktok"),
        ("https://www.instagram.com/reel/C-xyz123/?igsh=abc", "instagram"),
        ("https://instagram.com/reel/C-xyz123/", "instagram"),
        ("https://www.instagram.com/p/C-xyz123/", "instagram"),
        ("https://www.instagram.com/tv/C-xyz123/", "instagram"),
        ("https://instagr.am/p/C-xyz123/", "instagram"),
        ("https://www.instagram.com/share/reel/C-xyz123/", "instagram"),
        ("https://www.instagram.com/stories/username/1234567890/", "instagram"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=10s", "youtube"),
        ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share", "youtube"),
        ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "youtube"),
    ],
)
async def test_media_url_filter_platforms(url: str, platform_expected: str):
    filter_obj = MediaUrlFilter()

    # Standalone URL
    res = await filter_obj(create_message(text=url))
    assert res is not None and res is not False
    assert res["platform"] == platform_expected
    assert res["url"] == url
    assert res["media_url"] == url

    # URL surrounded by arbitrary text
    surrounded_text = f"Check this out! {url}. It is awesome!"
    res_surrounded = await filter_obj(create_message(text=surrounded_text))
    assert res_surrounded is not None and res_surrounded is not False
    assert res_surrounded["platform"] == platform_expected
    assert res_surrounded["url"] == url

    # In caption
    res_caption = await filter_obj(create_message(caption=f"Look here: {url}"))
    assert res_caption is not None and res_caption is not False
    assert res_caption["platform"] == platform_expected
    assert res_caption["url"] == url


@pytest.mark.asyncio
async def test_media_url_filter_specific_platform():
    tiktok_only_filter = MediaUrlFilter(Platform.TIKTOK)

    # TikTok URL matches
    tiktok_msg = create_message(text="https://vm.tiktok.com/ZM8vABC12/")
    res = await tiktok_only_filter(tiktok_msg)
    assert res == {
        "url": "https://vm.tiktok.com/ZM8vABC12/",
        "platform": "tiktok",
        "media_url": "https://vm.tiktok.com/ZM8vABC12/",
    }

    # YouTube URL should not match TikTok-only filter
    yt_msg = create_message(text="https://youtu.be/dQw4w9WgXcQ")
    assert await tiktok_only_filter(yt_msg) is False


@pytest.mark.asyncio
async def test_media_url_filter_non_media_urls():
    filter_obj = MediaUrlFilter()

    assert await filter_obj(create_message(text="https://google.com")) is False
    assert (
        await filter_obj(create_message(text="https://github.com/tmwrk34/tg-downloader")) is False
    )
    assert await filter_obj(create_message(text="https://example.org/index.html")) is False
    assert await filter_obj(create_message(text="Just some plain text without links")) is False
    assert await filter_obj(create_message(text=None, caption=None)) is False
