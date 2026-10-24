from typing import Any

from downloader.platforms.base import BasePlatformService
from filters.patterns import MEDIA_PATTERNS, Platform


class TikTokService(BasePlatformService):
    """Platform adapter for TikTok videos and photos."""

    @property
    def platform_name(self) -> str:
        return "tiktok"

    def is_supported(self, url: str) -> bool:
        pattern = MEDIA_PATTERNS.get(Platform.TIKTOK)
        return bool(pattern and pattern.search(url))

    def get_ytdl_options(self) -> dict[str, Any]:
        return {
            "format": "best[ext=mp4]/best",
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            },
        }
