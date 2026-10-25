from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from downloader.core import MediaDownloaderCore
from downloader.models import DownloadedMedia
from downloader.platforms.pinterest import (
    PinterestService,
    _shortlink_cache,
    extract_pin_id,
    resolve_pinterest_url,
)
from downloader.platforms.registry import PlatformServiceRegistry
from filters.patterns import MEDIA_PATTERNS, Platform
from services.cache import get_canonical_cache_key, normalize_media_url


def test_pinterest_pattern_matching():
    pattern = MEDIA_PATTERNS[Platform.PINTEREST]

    valid_urls = [
        "https://www.pinterest.com/pin/1234567890/",
        "https://pinterest.com/pin/1234567890",
        "https://co.pinterest.com/pin/9988776655/",
        "https://www.pinterest.co.uk/pin/1122334455/",
        "https://pin.it/7abcXYZ",
        "https://www.pin.it/abc1234",
        "https://www.pinterest.com/pin/aesthetic-decor-ideas--1234567890/",
        "Check this out: https://pin.it/39v8aBc for inspiration!",
    ]

    for url in valid_urls:
        match = pattern.search(url)
        assert match is not None, f"URL should match: {url}"

    invalid_urls = [
        "https://www.pinterest.com/user/profile/",
        "https://www.pinterest.com/search/pins/?q=nature",
        "https://instagram.com/p/12345/",
        "https://youtube.com/watch?v=123",
    ]

    for url in invalid_urls:
        match = pattern.search(url)
        assert match is None, f"URL should NOT match: {url}"


def test_pinterest_extract_pin_id():
    assert extract_pin_id("https://www.pinterest.com/pin/1234567890/") == "1234567890"
    assert extract_pin_id("https://pinterest.com/pin/987654321") == "987654321"
    assert (
        extract_pin_id("https://www.pinterest.com/pin/cozy-bedroom-ideas--5544332211/")
        == "5544332211"
    )
    assert extract_pin_id("https://pin.it/abcXYZ") is None


def test_resolve_pinterest_url_and_lru_caching():
    _shortlink_cache.clear()

    # Non-shortlink returns immediately
    reg_url = "https://www.pinterest.com/pin/1234567890/"
    assert resolve_pinterest_url(reg_url) == reg_url

    # Mock shortlink redirect
    mock_resp = MagicMock()
    mock_resp.geturl.return_value = "https://www.pinterest.com/pin/888777666/"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        res1 = resolve_pinterest_url("https://pin.it/testShort1")
        res2 = resolve_pinterest_url("https://pin.it/testShort1")

        assert res1 == "https://www.pinterest.com/pin/888777666/"
        assert res2 == "https://www.pinterest.com/pin/888777666/"
        # Should only call network once due to LRU cache
        assert mock_open.call_count == 1

    # Network error fallback
    with patch("urllib.request.urlopen", side_effect=Exception("network err")):
        fallback = resolve_pinterest_url("https://pin.it/errShort1")
        assert fallback == "https://pin.it/errShort1"


def test_pinterest_url_normalization():
    norm1 = normalize_media_url("https://www.pinterest.com/pin/1234567890/?invite_code=abc")
    assert norm1 == "pinterest:pin:1234567890"

    norm2 = normalize_media_url("https://www.pinterest.com/pin/cool-title--987654321/")
    assert norm2 == "pinterest:pin:987654321"

    norm_short = normalize_media_url("https://pin.it/xyz123")
    assert norm_short == "pinterest:shortlink:xyz123"


@pytest.mark.asyncio
async def test_pinterest_canonical_cache_key():
    with patch(
        "services.cache.resolve_pinterest_url",
        return_value="https://www.pinterest.com/pin/1234567890/",
    ):
        key = await get_canonical_cache_key("https://pin.it/xyz123", platform="pinterest")
        assert key == "pinterest:pin:1234567890"


def test_platform_registry_supports_pinterest(tmp_path: Path):
    core = MediaDownloaderCore(download_dir=tmp_path)
    registry = PlatformServiceRegistry(core)

    svc_name = registry.get_service_by_name("pinterest")
    assert svc_name is not None
    assert isinstance(svc_name, PinterestService)

    svc_url = registry.get_service_for_url("https://www.pinterest.com/pin/1234567890/")
    assert svc_url is not None
    assert isinstance(svc_url, PinterestService)

    svc_short = registry.get_service_for_url("https://pin.it/abc1234")
    assert svc_short is not None
    assert isinstance(svc_short, PinterestService)


@pytest.mark.asyncio
async def test_pinterest_download_ytdl_path(tmp_path: Path):
    core = MediaDownloaderCore(download_dir=tmp_path)
    service = PinterestService(core)

    fake_downloaded = DownloadedMedia(
        filepath=tmp_path / "pin_video.mp4",
        title="Pinterest Video Pin",
        filesize=1024,
    )

    with (
        patch(
            "downloader.platforms.pinterest.resolve_pinterest_url",
            return_value="https://www.pinterest.com/pin/111/",
        ),
        patch(
            "downloader.platforms.base.BasePlatformService.download", new_callable=AsyncMock
        ) as mock_super,
    ):
        mock_super.return_value = fake_downloaded
        res = await service.download("https://pin.it/shortVideo")
        assert res.title == "Pinterest Video Pin"
        assert res.filepath == tmp_path / "pin_video.mp4"


@pytest.mark.asyncio
async def test_pinterest_download_direct_api_photo_fallback(tmp_path: Path):
    core = MediaDownloaderCore(download_dir=tmp_path)
    service = PinterestService(core)

    pin_id = "555666777"
    raw_api_data = {
        "title": "Aesthetic Room Photo",
        "grid_title": "Aesthetic Room",
        "closeup_attribution": {"full_name": "Aesthetic Creator"},
        "images": {
            "orig": {
                "url": "https://i.pinimg.com/originals/fake_orig.jpg",
                "width": 1080,
                "height": 1920,
            }
        },
    }

    with (
        patch(
            "downloader.platforms.pinterest.resolve_pinterest_url",
            return_value=f"https://www.pinterest.com/pin/{pin_id}/",
        ),
        patch(
            "downloader.platforms.base.BasePlatformService.download",
            side_effect=Exception("yt-dlp no video formats"),
        ),
        patch.object(service, "_fetch_pin_data_sync", return_value=raw_api_data),
        patch("downloader.platforms.pinterest.download_url_to_file") as mock_dl_file,
    ):

        def fake_dl(url, path):
            Path(path).write_bytes(b"fake_image_bytes")

        mock_dl_file.side_effect = fake_dl

        res = await service.download("https://pin.it/photoPin")
        assert res.title == "Aesthetic Room Photo"
        assert res.artist == "Aesthetic Creator"
        assert res.is_gallery is False
        assert res.filepath.suffix == ".jpg"
        assert res.filepath.exists()


@pytest.mark.asyncio
async def test_pinterest_download_direct_api_idea_pin_carousel(tmp_path: Path):
    core = MediaDownloaderCore(download_dir=tmp_path)
    service = PinterestService(core)

    pin_id = "999888777"
    raw_api_data = {
        "title": "Outfit Ideas Collection",
        "closeup_attribution": {"full_name": "Fashion Blogger"},
        "story_pin_data": {
            "pages": [
                {
                    "image": {
                        "images": {
                            "originals": {
                                "url": "https://i.pinimg.com/originals/slide1.jpg",
                                "width": 800,
                                "height": 1200,
                            }
                        }
                    }
                },
                {
                    "image": {
                        "images": {
                            "originals": {
                                "url": "https://i.pinimg.com/originals/slide2.jpg",
                                "width": 800,
                                "height": 1200,
                            }
                        }
                    }
                },
            ]
        },
    }

    with (
        patch(
            "downloader.platforms.pinterest.resolve_pinterest_url",
            return_value=f"https://www.pinterest.com/pin/{pin_id}/",
        ),
        patch(
            "downloader.platforms.base.BasePlatformService.download",
            side_effect=Exception("yt-dlp failed"),
        ),
        patch.object(service, "_fetch_pin_data_sync", return_value=raw_api_data),
        patch("downloader.platforms.pinterest.download_url_to_file") as mock_dl_file,
    ):

        def fake_dl(url, path):
            Path(path).write_bytes(b"slide_bytes")

        mock_dl_file.side_effect = fake_dl

        res = await service.download(f"https://www.pinterest.com/pin/{pin_id}/")
        assert res.title == "Outfit Ideas Collection"
        assert res.is_gallery is True
        assert len(res.gallery_items) == 2
        assert res.gallery_items[0][0] == "photo"
        assert res.gallery_items[1][0] == "photo"
