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
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "ignore_no_formats_error": True,
        }
