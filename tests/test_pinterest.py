from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from downloader.core import MediaDownloaderCore
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
        assert mock_open.call_count == 1

    # Network error fallback
    with patch("urllib.request.urlopen", side_effect=Exception("network err")):
        fallback = resolve_pinterest_url("https://pin.it/errShort1")
        assert fallback == "https://pin.it/errShort1"


def test_pinterest_url_normalization():
    norm1 = normalize_media_url("https://www.pinterest.com/pin/1234567890/?invite_code=abc")
    assert norm1 == "pinterest:v3:pin:1234567890"

    norm2 = normalize_media_url("https://www.pinterest.com/pin/cool-title--987654321/")
    assert norm2 == "pinterest:v3:pin:987654321"

    norm_short = normalize_media_url("https://pin.it/xyz123")
    assert norm_short == "pinterest:v3:shortlink:xyz123"


@pytest.mark.asyncio
async def test_pinterest_canonical_cache_key():
    with patch(
        "services.cache.resolve_pinterest_url",
        return_value="https://www.pinterest.com/pin/1234567890/",
    ):
        key = await get_canonical_cache_key("https://pin.it/xyz123", platform="pinterest")
        assert key == "pinterest:v3:pin:1234567890"


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
async def test_pinterest_download_carousel_slots_10_photos(tmp_path: Path):
    core = MediaDownloaderCore(download_dir=tmp_path)
    service = PinterestService(core)

    pin_id = "1055531231435344334"
    # Construct 10 carousel slots matching Pinterest Relay format
    slots = [
        {
            "images_1200x": {"url": f"https://i.pinimg.com/1200x/img_{i}.jpg"},
        }
        for i in range(10)
    ]
    raw_api_data = {
        "title": "10 Photo Collage",
        "closeup_attribution": {"full_name": "Collage Creator"},
        "carouselData": {"carouselSlots": slots},
    }

    with (
        patch(
            "downloader.platforms.pinterest.resolve_pinterest_url",
            return_value=f"https://www.pinterest.com/pin/{pin_id}/",
        ),
        patch.object(service, "_fetch_pin_data_sync", return_value=raw_api_data),
        patch("downloader.platforms.pinterest.download_url_to_file") as mock_dl_file,
    ):

        def fake_dl(url, path):
            Path(path).write_bytes(b"image_bytes")

        mock_dl_file.side_effect = fake_dl

        res = await service.download("https://pin.it/1BHRnKwgY")
        assert res.title == "10 Photo Collage"
        assert res.artist == "Collage Creator"
        assert res.is_gallery is True
        assert len(res.gallery_items) == 10
        assert all(item[0] == "photo" for item in res.gallery_items)
        assert mock_dl_file.call_count == 10


@pytest.mark.asyncio
async def test_pinterest_download_single_photo(tmp_path: Path):
    core = MediaDownloaderCore(download_dir=tmp_path)
    service = PinterestService(core)

    pin_id = "1105352302312686389"
    raw_api_data = {
        "title": "Single Aesthetic Photo",
        "closeup_attribution": {"full_name": "Aesthetic Creator"},
        "images_orig": {"url": "https://i.pinimg.com/originals/photo1.jpg"},
    }

    with (
        patch(
            "downloader.platforms.pinterest.resolve_pinterest_url",
            return_value=f"https://www.pinterest.com/pin/{pin_id}/",
        ),
        patch.object(service, "_fetch_pin_data_sync", return_value=raw_api_data),
        patch("downloader.platforms.pinterest.download_url_to_file") as mock_dl_file,
    ):

        def fake_dl(url, path):
            Path(path).write_bytes(b"photo_bytes")

        mock_dl_file.side_effect = fake_dl

        res = await service.download("https://pin.it/ccPRWi6Xl")
        assert res.title == "Single Aesthetic Photo"
        assert res.is_gallery is False
        assert res.filepath.name == f"{pin_id}.jpg"
        assert res.filepath.exists()


@pytest.mark.asyncio
async def test_pinterest_download_single_video_from_story_pin(tmp_path: Path):
    core = MediaDownloaderCore(download_dir=tmp_path)
    service = PinterestService(core)

    pin_id = "876864989963745171"
    raw_api_data = {
        "title": "Aesthetic Video Pin",
        "closeup_attribution": {"full_name": "Video Creator"},
        "storyPinData": {
            "pages": [
                {
                    "blocks": [
                        {
                            "videoDataV2": {
                                "videoList720P": {
                                    "v720P": {
                                        "url": "https://v1.pinimg.com/videos/iht/expMp4/test_720w.mp4",
                                        "width": 1440,
                                        "height": 1440,
                                        "duration": 16367,
                                    }
                                }
                            }
                        }
                    ]
                }
            ]
        },
    }

    with (
        patch(
            "downloader.platforms.pinterest.resolve_pinterest_url",
            return_value=f"https://www.pinterest.com/pin/{pin_id}/",
        ),
        patch.object(service, "_fetch_pin_data_sync", return_value=raw_api_data),
        patch("downloader.platforms.pinterest.download_url_to_file") as mock_dl_file,
    ):

        def fake_dl(url, path):
            Path(path).write_bytes(b"video_bytes")

        mock_dl_file.side_effect = fake_dl

        res = await service.download("https://pin.it/1rhkhuVRO")
        assert res.title == "Aesthetic Video Pin"
        assert res.artist == "Video Creator"
        assert res.is_gallery is False
        assert res.filepath.name == f"{pin_id}.mp4"
        assert res.duration == 16
        assert res.width == 1440
        assert res.height == 1440
        assert res.filepath.exists()


def test_pinterest_fetch_pin_data_from_relay_html(tmp_path: Path):
    core = MediaDownloaderCore(download_dir=tmp_path)
    service = PinterestService(core)

    mock_html = """
    <html>
      <head></head>
      <body>
        <script>
          window.__PWS_RELAY_REGISTER_COMPLETED_REQUEST__(
            "%7Bquery%7D",
            {
              "data": {
                "v3GetPinQueryv2": {
                  "data": {
                    "title": "Relay Extracted Pin",
                    "images_orig": {"url": "https://i.pinimg.com/originals/test.jpg"}
                  }
                }
              }
            }
          );
        </script>
      </body>
    </html>
    """

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_html.encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        data = service._fetch_pin_data_sync("12345")
        assert data.get("title") == "Relay Extracted Pin"
        assert data.get("images_orig", {}).get("url") == "https://i.pinimg.com/originals/test.jpg"
