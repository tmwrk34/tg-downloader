from typing import Any

from downloader.platforms.base import BasePlatformService
from filters.patterns import MEDIA_PATTERNS, Platform


class YouTubeService(BasePlatformService):
    """Platform adapter for YouTube Videos, Shorts, and Audio."""

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

    def get_format_options(self, quality: str) -> dict[str, Any]:
        """Return yt-dlp options tailored for specific video resolution or audio extraction."""
        if quality == "audio":
            return {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }
        elif quality == "1080":
            return {
                "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080][ext=mp4]/best",
                "merge_output_format": "mp4",
            }
        elif quality == "720":
            return {
                "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720][ext=mp4]/best",
                "merge_output_format": "mp4",
            }
        elif quality == "360":
            return {
                "format": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=360]+bestaudio/best[height<=360][ext=mp4]/best",
                "merge_output_format": "mp4",
            }
        return self.get_ytdl_options()
