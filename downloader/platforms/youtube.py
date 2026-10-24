from typing import Any

from downloader.platforms.base import BasePlatformService
from filters.patterns import MEDIA_PATTERNS, Platform


class YouTubeService(BasePlatformService):
    """Platform adapter for YouTube Videos and Shorts."""

    @property
    def platform_name(self) -> str:
        return "youtube"

    def is_supported(self, url: str) -> bool:
        pattern = MEDIA_PATTERNS.get(Platform.YOUTUBE)
        return bool(pattern and pattern.search(url))

    def get_ytdl_options(self) -> dict[str, Any]:
        return {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best",
            "merge_output_format": "mp4",
        }
