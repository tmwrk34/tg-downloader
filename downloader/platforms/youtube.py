from typing import Any

from downloader.models import MediaMetadata
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
                "outtmpl": "%(artist,uploader)s - %(track,title)s.%(ext)s",
                "writethumbnail": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    },
                    {
                        "key": "FFmpegMetadata",
                        "add_metadata": True,
                    },
                    {
                        "key": "EmbedThumbnail",
                    },
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

    def estimate_format_size(self, meta: MediaMetadata, quality: str) -> int | None:
        """Estimate payload size in bytes for a given quality using format list or duration bitrates."""
        if quality == "audio":
            if meta.duration:
                # 192 kbps audio estimate
                return int(192 * 1024 / 8 * meta.duration)
            return None

        target_h = 1080 if quality == "1080" else (720 if quality == "720" else 360)

        # Check if matching video format reports exact/approximate filesize
        matching_v_sizes = [
            f.get("filesize") or f.get("filesize_approx")
            for f in meta.formats
            if f.get("height") and f.get("height") <= target_h and f.get("vcodec") != "none"
        ]
        valid_sizes = [s for s in matching_v_sizes if s]
        if valid_sizes:
            v_size = max(valid_sizes)
            a_size = int(128 * 1024 / 8 * (meta.duration or 0))
            return v_size + a_size

        # Fallback based on typical resolution bitrates
        bitrates_kbps = {"1080": 3500, "720": 1500, "360": 500}
        if meta.duration:
            total_kbps = bitrates_kbps.get(quality, 1500) + 128
            return int(total_kbps * 1000 / 8 * meta.duration)

        return None
