import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from downloader import (
    BasePlatformService,
    DownloadedMedia,
    InstagramService,
    MediaDownloaderCore,
    MediaMetadata,
    PlatformServiceRegistry,
    TikTokService,
    YouTubeService,
)
from downloader.core import clean_artist_tag
from downloader.platforms.tiktok import resolve_tiktok_url


@pytest.fixture
def temp_download_dir(tmp_path: Path) -> Path:
    dl_dir = tmp_path / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    return dl_dir


@pytest.fixture
def core_downloader(temp_download_dir: Path) -> MediaDownloaderCore:
    return MediaDownloaderCore(download_dir=temp_download_dir)


@pytest.mark.asyncio
async def test_downloaded_media_context_manager_cleanup_on_success(tmp_path: Path):
    job_dir = tmp_path / "job_success"
    job_dir.mkdir()
    media_file = job_dir / "test.mp4"
    media_file.write_bytes(b"dummy video content")

    media = DownloadedMedia(
        filepath=media_file,
        title="Test Title",
        temp_dir=job_dir,
    )

    assert media_file.exists()
    assert job_dir.exists()

    async with media:
        assert media.filepath.exists()
        assert media.title == "Test Title"

    # Must be deleted after exiting context block
    assert not media_file.exists()
    assert not job_dir.exists()


@pytest.mark.asyncio
async def test_downloaded_media_context_manager_cleanup_on_exception(tmp_path: Path):
    job_dir = tmp_path / "job_error"
    job_dir.mkdir()
    media_file = job_dir / "test.mp4"
    media_file.write_bytes(b"dummy video content")

    media = DownloadedMedia(
        filepath=media_file,
        title="Test Error",
        temp_dir=job_dir,
    )

    with pytest.raises(RuntimeError, match="Upload failed"):
        async with media:
            assert media_file.exists()
            raise RuntimeError("Upload failed")

    # Must be deterministically deleted even after an exception
    assert not media_file.exists()
    assert not job_dir.exists()


@pytest.mark.asyncio
async def test_core_cleanup_on_download_failure(core_downloader: MediaDownloaderCore):
    # Mock yt_dlp to raise an exception during extract_info
    with patch("downloader.core.yt_dlp.YoutubeDL") as mock_ydl_cls:
        instance = MagicMock()
        instance.extract_info.side_effect = Exception("Download failed in yt-dlp")
        mock_ydl_cls.return_value.__enter__.return_value = instance

        with pytest.raises(Exception, match="Download failed in yt-dlp"):
            await core_downloader.download("https://example.com/failed")

        # Check that no lingering job_* folders remain in download_dir
        remaining_dirs = list(core_downloader.download_dir.glob("job_*"))
        assert len(remaining_dirs) == 0


@pytest.mark.asyncio
async def test_core_extract_metadata(core_downloader: MediaDownloaderCore):
    fake_info = {
        "title": "Sample Video",
        "duration": 42,
        "thumbnail": "https://example.com/thumb.jpg",
        "width": 1920,
        "height": 1080,
        "filesize": 1234567,
        "extractor": "youtube",
        "formats": [{"format_id": "18", "ext": "mp4"}],
    }

    with patch("downloader.core.yt_dlp.YoutubeDL") as mock_ydl_cls:
        instance = MagicMock()
        instance.extract_info.return_value = fake_info
        mock_ydl_cls.return_value.__enter__.return_value = instance

        meta = await core_downloader.extract_metadata("https://youtu.be/dQw4w9WgXcQ")

        assert isinstance(meta, MediaMetadata)
        assert meta.title == "Sample Video"
        assert meta.duration == 42
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.filesize == 1234567
        assert meta.extractor == "youtube"


@pytest.mark.asyncio
async def test_core_download_success(core_downloader: MediaDownloaderCore):
    def fake_extract_info(url, download=True):
        # Create the dummy downloaded file in the target directory
        outtmpl = core_downloader.download_dir
        job_dirs = list(outtmpl.glob("job_*"))
        target_dir = job_dirs[0] if job_dirs else outtmpl
        video_file = target_dir / "sample.mp4"
        video_file.write_bytes(b"dummy-mp4-payload")
        return {
            "title": "Downloaded Video",
            "duration": 60,
            "width": 1280,
            "height": 720,
            "_filename": str(video_file),
        }

    with patch("downloader.core.yt_dlp.YoutubeDL") as mock_ydl_cls:
        instance = MagicMock()
        instance.extract_info.side_effect = fake_extract_info
        instance.prepare_filename.side_effect = lambda info: info["_filename"]
        mock_ydl_cls.return_value.__enter__.return_value = instance

        media = await core_downloader.download("https://www.tiktok.com/@user/video/1234567890")

        assert isinstance(media, DownloadedMedia)
        assert media.title == "Downloaded Video"
        assert media.filepath.exists()
        assert media.filesize == len(b"dummy-mp4-payload")

        # Cleanup
        media.cleanup()
        assert not media.filepath.exists()


@pytest.mark.asyncio
async def test_non_blocking_event_loop_concurrency(core_downloader: MediaDownloaderCore):
    """Ensure core downloader executes non-blocking I/O without freezing asyncio."""
    slow_done = False

    def fake_slow_extract(url, download=False):
        import time

        time.sleep(0.3)
        return {"title": "Slow Video"}

    async def background_counter():
        ticks = 0
        while not slow_done:
            await asyncio.sleep(0.05)
            ticks += 1
        return ticks

    with patch("downloader.core.yt_dlp.YoutubeDL") as mock_ydl_cls:
        instance = MagicMock()
        instance.extract_info.side_effect = fake_slow_extract
        mock_ydl_cls.return_value.__enter__.return_value = instance

        counter_task = asyncio.create_task(background_counter())
        await core_downloader.extract_metadata("https://youtube.com/watch?v=slow")
        slow_done = True
        ticks = await counter_task

        # Event loop ticked multiple times during the sync sleep in thread
        assert ticks >= 2


def test_platform_service_adapters(core_downloader: MediaDownloaderCore):
    tiktok = TikTokService(core_downloader)
    insta = InstagramService(core_downloader)
    yt = YouTubeService(core_downloader)

    assert tiktok.platform_name == "tiktok"
    assert tiktok.is_supported("https://vm.tiktok.com/ZM8vABC12/")
    assert not tiktok.is_supported("https://youtube.com/watch?v=123")
    assert "format" in tiktok.get_ytdl_options()

    assert insta.platform_name == "instagram"
    assert insta.is_supported("https://www.instagram.com/reel/C-xyz123/")
    assert not insta.is_supported("https://vm.tiktok.com/ZM8vABC12/")

    assert yt.platform_name == "youtube"
    assert yt.is_supported("https://youtu.be/dQw4w9WgXcQ")
    assert not yt.is_supported("https://instagram.com/reel/C-xyz123/")

    # YouTube format and audio options
    audio_opts = yt.get_format_options("audio")
    assert "postprocessors" in audio_opts
    assert audio_opts["postprocessors"][0]["preferredcodec"] == "mp3"
    assert any(p.get("key") == "EmbedThumbnail" for p in audio_opts["postprocessors"])
    assert any(p.get("key") == "FFmpegMetadata" for p in audio_opts["postprocessors"])

    res_1080 = yt.get_format_options("1080")
    assert "1080" in res_1080["format"]
    assert res_1080["merge_output_format"] == "mp4"

    res_720 = yt.get_format_options("720")
    assert "720" in res_720["format"]

    res_360 = yt.get_format_options("360")
    assert "360" in res_360["format"]

    # Size estimation tests
    meta = MediaMetadata(
        title="Test Track",
        duration=180,  # 3 minutes
        formats=[
            {"format_id": "1", "height": 1080, "vcodec": "avc1", "filesize": 40_000_000},
            {"format_id": "2", "height": 720, "vcodec": "avc1", "filesize": 20_000_000},
        ],
    )
    est_1080 = yt.estimate_format_size(meta, "1080")
    assert est_1080 is not None
    assert est_1080 > 40_000_000

    est_audio = yt.estimate_format_size(meta, "audio")
    assert est_audio is not None
    assert est_audio == int(192 * 1024 / 8 * 180)


def test_platform_service_extensibility(core_downloader: MediaDownloaderCore):
    """Verify that adding a new platform requires only creating a new service class."""

    class CustomPlatformService(BasePlatformService):
        @property
        def platform_name(self) -> str:
            return "custom"

        def is_supported(self, url: str) -> bool:
            return "custom.platform" in url

        def get_ytdl_options(self) -> dict:
            return {"format": "custom_best"}

    registry = PlatformServiceRegistry(core_downloader)

    # Initial state does not know custom platform
    test_url = "https://custom.platform/video/42"
    assert registry.get_service_for_url(test_url) is None

    # Register custom platform without altering core engine
    custom_service = CustomPlatformService(core_downloader)
    registry.register(custom_service)

    matched = registry.get_service_for_url(test_url)
    assert matched is not None
    assert matched.platform_name == "custom"
    assert matched.get_ytdl_options() == {"format": "custom_best"}
    assert registry.get_service_by_name("custom") is custom_service


def test_clean_artist_tag():
    assert (
        clean_artist_tag(
            "Atari Teenage Riot, Atari Teenage Riot",
            ["Atari Teenage Riot", "Atari Teenage Riot"],
        )
        == "Atari Teenage Riot"
    )
    assert (
        clean_artist_tag(
            "Daft Punk, Pharrell Williams",
            ["Daft Punk", "Pharrell Williams"],
        )
        == "Daft Punk, Pharrell Williams"
    )
    assert clean_artist_tag("Band Name, Band Name") == "Band Name"
    assert clean_artist_tag(None, None, "The Prodigy - Topic") == "The Prodigy"
    assert clean_artist_tag(None, None, "Normal Uploader") == "Normal Uploader"
    assert clean_artist_tag(None, None, None) is None


def test_youtube_extract_video_id():
    assert (
        YouTubeService.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )
    assert YouTubeService.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert (
        YouTubeService.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )
    assert (
        YouTubeService.extract_video_id("https://music.youtube.com/watch?v=dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )
    assert YouTubeService.extract_video_id("https://example.com/not-youtube") is None


@pytest.mark.asyncio
async def test_downloaded_media_gallery_cleanup(tmp_path: Path):
    job_dir = tmp_path / "job_gallery"
    job_dir.mkdir()
    photo1 = job_dir / "p1.jpg"
    video1 = job_dir / "v1.mp4"
    photo1.write_bytes(b"image")
    video1.write_bytes(b"video")

    media = DownloadedMedia(
        filepath=photo1,
        title="Gallery Title",
        temp_dir=job_dir,
        is_gallery=True,
        gallery_items=[("photo", photo1), ("video", video1)],
    )

    assert photo1.exists()
    assert video1.exists()

    async with media:
        assert media.is_gallery
        assert len(media.gallery_items) == 2

    assert not photo1.exists()
    assert not video1.exists()
    assert not job_dir.exists()


@pytest.mark.asyncio
async def test_core_download_gallery_multi_item(
    core_downloader: MediaDownloaderCore, tmp_path: Path
):
    fake_info = {
        "title": "Post Carousel",
        "entries": [
            {
                "id": "item1",
                "formats": [],
                "thumbnails": [{"url": "https://example.com/item1.jpg", "preference": 1}],
            },
            {
                "id": "item2",
                "formats": [
                    {"url": "https://example.com/item2.mp4", "vcodec": "h264", "height": 720}
                ],
                "thumbnails": [],
            },
        ],
    }

    with (
        patch("downloader.core.yt_dlp.YoutubeDL") as mock_ydl_cls,
        patch("downloader.core.download_url_to_file") as mock_dl,
    ):
        instance = MagicMock()
        instance.extract_info.return_value = fake_info
        mock_ydl_cls.return_value.__enter__.return_value = instance

        # Simulate writing dummy bytes on download_url_to_file
        def fake_download_url(url, dest_path):
            dest_path.write_bytes(b"dummy_content")

        mock_dl.side_effect = fake_download_url

        downloaded = await core_downloader.download("https://instagram.com/p/carousel123")
        assert downloaded.is_gallery is True
        assert len(downloaded.gallery_items) == 2
        assert downloaded.gallery_items[0][0] == "photo"
        assert downloaded.gallery_items[1][0] == "video"
        assert downloaded.filesize > 0
        downloaded.cleanup()


@pytest.mark.asyncio
async def test_core_download_single_photo(core_downloader: MediaDownloaderCore, tmp_path: Path):
    fake_info = {
        "id": "single_photo_1",
        "title": "Solo Photo",
        "formats": [],
        "thumbnails": [{"url": "https://example.com/photo.jpg", "preference": 1}],
    }

    with (
        patch("downloader.core.yt_dlp.YoutubeDL") as mock_ydl_cls,
        patch("downloader.core.download_url_to_file") as mock_dl,
    ):
        instance = MagicMock()
        instance.extract_info.return_value = fake_info
        mock_ydl_cls.return_value.__enter__.return_value = instance

        def fake_download_url(url, dest_path):
            dest_path.write_bytes(b"photo_content")

        mock_dl.side_effect = fake_download_url

        downloaded = await core_downloader.download("https://instagram.com/p/singlephoto")
        assert downloaded.is_gallery is False
        assert downloaded.filepath.name.endswith(".jpg")
        assert downloaded.filesize > 0
        downloaded.cleanup()


@pytest.mark.asyncio
async def test_send_downloaded_media_gallery_rich_slideshow(tmp_path: Path):
    from unittest.mock import AsyncMock

    from bot import send_downloaded_media

    items = []
    for i in range(14):
        p = tmp_path / f"img_{i:02d}.jpg"
        p.write_bytes(b"data")
        items.append(("photo", p))

    downloaded = DownloadedMedia(
        filepath=items[0][1],
        title="Big Carousel",
        filesize=1000,
        temp_dir=tmp_path,
        is_gallery=True,
        gallery_items=items,
    )

    mock_msg = MagicMock()
    mock_msg.reply_rich = AsyncMock(return_value=True)
    mock_msg.reply_media_group = AsyncMock(return_value=True)

    sent = await send_downloaded_media(mock_msg, downloaded)
    assert sent is True
    # Successfully sent via reply_rich (2 chunks: 10 and 4)
    assert mock_msg.reply_rich.call_count == 2
    assert mock_msg.reply_media_group.call_count == 0


@pytest.mark.asyncio
async def test_send_downloaded_media_gallery_fallback_to_media_group(tmp_path: Path):
    from unittest.mock import AsyncMock

    from bot import send_downloaded_media

    items = []
    for i in range(14):
        p = tmp_path / f"img_{i:02d}.jpg"
        p.write_bytes(b"data")
        items.append(("photo", p))

    downloaded = DownloadedMedia(
        filepath=items[0][1],
        title="Big Carousel",
        filesize=1000,
        temp_dir=tmp_path,
        is_gallery=True,
        gallery_items=items,
    )

    mock_msg = MagicMock()
    mock_msg.reply_rich = AsyncMock(side_effect=Exception("Rich message method not supported"))
    mock_msg.reply_media_group = AsyncMock(return_value=True)

    sent = await send_downloaded_media(mock_msg, downloaded)
    assert sent is True
    # Failed on reply_rich, fell back to reply_media_group
    assert mock_msg.reply_rich.call_count == 2
    assert mock_msg.reply_media_group.call_count == 2
    first_call_media = mock_msg.reply_media_group.call_args_list[0].kwargs["media"]
    second_call_media = mock_msg.reply_media_group.call_args_list[1].kwargs["media"]
    assert len(first_call_media) == 10
    assert len(second_call_media) == 4


@pytest.mark.asyncio
async def test_tiktok_photo_slideshow_download(
    core_downloader: MediaDownloaderCore, tmp_path: Path
):
    svc = TikTokService(core_downloader)

    fake_video_data = {
        "desc": "Funny Memes",
        "author": {"nickname": "MemeLord"},
        "imagePost": {
            "images": [
                {"imageURL": {"urlList": ["https://example.com/slide1.jpg"]}},
                {"imageURL": {"urlList": ["https://example.com/slide2.jpg"]}},
            ]
        },
        "music": {"playUrl": "https://example.com/audio.mp3"},
    }

    with (
        patch("downloader.platforms.tiktok.TikTokIE._extract_web_data_and_status") as mock_extract,
        patch("downloader.platforms.tiktok.download_url_to_file") as mock_dl,
    ):
        mock_extract.return_value = (fake_video_data, 0)

        def fake_dl(url, dest_path):
            dest_path.write_bytes(b"content")

        mock_dl.side_effect = fake_dl

        url = "https://www.tiktok.com/@tow1nner/photo/7650410187165093141"
        downloaded = await svc.download(url)
        assert downloaded.is_gallery is True
        assert len(downloaded.gallery_items) == 2
        assert downloaded.gallery_items[0][0] == "photo"
        assert downloaded.extra_audio_path is not None
        assert downloaded.artist == "MemeLord"
        downloaded.cleanup()


@pytest.mark.asyncio
async def test_tiktok_photo_slideshow_metadata(
    core_downloader: MediaDownloaderCore, tmp_path: Path
):
    svc = TikTokService(core_downloader)

    fake_video_data = {
        "desc": "Photo album",
        "author": {"uniqueId": "user123"},
        "imagePost": {
            "images": [
                {"imageURL": {"urlList": ["https://example.com/slide1.jpg"]}},
                {"imageURL": {"urlList": ["https://example.com/slide2.jpg"]}},
                {"imageURL": {"urlList": ["https://example.com/slide3.jpg"]}},
            ]
        },
    }

    with patch("downloader.platforms.tiktok.TikTokIE._extract_web_data_and_status") as mock_extract:
        mock_extract.return_value = (fake_video_data, 0)

        url = "https://www.tiktok.com/@user/photo/12345"
        meta = await svc.extract_metadata(url)
        assert meta.is_gallery is True
        assert meta.gallery_count == 3
        assert meta.artist == "user123"


def test_resolve_tiktok_url():
    # Non-shortlink URL should return immediately without network call
    reg_url = "https://www.tiktok.com/@user/video/12345"
    assert resolve_tiktok_url(reg_url) == reg_url

    # Mock shortlink redirect
    mock_resp = MagicMock()
    mock_resp.geturl.return_value = "https://www.tiktok.com/@roller_055/photo/7700307198781492488"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = resolve_tiktok_url("https://vt.tiktok.com/ZSgxuBW5a/")
        assert res == "https://www.tiktok.com/@roller_055/photo/7700307198781492488"

    # Exception should fall back to original URL
    with patch("urllib.request.urlopen", side_effect=Exception("network err")):
        fallback = resolve_tiktok_url("https://vt.tiktok.com/ZSgxuBW5a/")
        assert fallback == "https://vt.tiktok.com/ZSgxuBW5a/"


@pytest.mark.asyncio
async def test_tiktok_shortlink_routes_to_photo_slideshow(
    core_downloader: MediaDownloaderCore, tmp_path: Path
):
    svc = TikTokService(core_downloader)

    resolved_photo_url = "https://www.tiktok.com/@roller_055/photo/7700307198781492488"
    fake_downloaded = DownloadedMedia(
        filepath=tmp_path / "slide_00.jpg",
        title="Photo Slideshow",
        is_gallery=True,
        gallery_items=[("photo", tmp_path / "slide_00.jpg")],
    )

    with (
        patch(
            "downloader.platforms.tiktok.resolve_tiktok_url", return_value=resolved_photo_url
        ) as mock_resolve,
        patch.object(
            svc, "_download_photo_slideshow_sync", return_value=fake_downloaded
        ) as mock_ss,
    ):
        res = await svc.download("https://vt.tiktok.com/ZSgxuBW5a/")
        mock_resolve.assert_called_once_with("https://vt.tiktok.com/ZSgxuBW5a/")
        mock_ss.assert_called_once_with(resolved_photo_url)
        assert res.is_gallery is True


def test_core_cookie_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    core = MediaDownloaderCore(download_dir=tmp_path)

    # 1. Test COOKIES_FILE env var
    cookie_file = tmp_path / "custom_cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File", encoding="utf-8")
    monkeypatch.setenv("COOKIES_FILE", str(cookie_file))
    assert core._get_cookie_file() == str(cookie_file.resolve())
    opts = core._build_ytdl_base_opts()
    assert opts.get("cookiefile") == str(cookie_file.resolve())

    # 2. Test INSTAGRAM_COOKIES env var writing to /tmp/cookies.txt
    monkeypatch.delenv("COOKIES_FILE")
    monkeypatch.setenv("INSTAGRAM_COOKIES", "# IG Cookies Raw")
    cookie_path = core._get_cookie_file()
    assert cookie_path is not None
    assert Path(cookie_path).read_text(encoding="utf-8") == "# IG Cookies Raw"
