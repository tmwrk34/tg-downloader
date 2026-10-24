from typing import Any

from downloader.platforms.base import BasePlatformService
from filters.patterns import MEDIA_PATTERNS, Platform


class InstagramService(BasePlatformService):
    """Platform adapter for Instagram Reels, Posts, and Stories."""

    @property
    def platform_name(self) -> str:
        return "instagram"

    def is_supported(self, url: str) -> bool:
        pattern = MEDIA_PATTERNS.get(Platform.INSTAGRAM)
        return bool(pattern and pattern.search(url))

    def get_ytdl_options(self) -> dict[str, Any]:
        return {
            "format": "best[ext=mp4]/best",
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
                ),
            },
        }
