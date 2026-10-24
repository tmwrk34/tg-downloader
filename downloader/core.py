import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import yt_dlp

from downloader.models import DownloadedMedia, MediaMetadata

logger = logging.getLogger(__name__)


class MediaDownloaderCore:
    """Core asynchronous media downloader using yt-dlp."""

    def __init__(self, download_dir: Path | str = "./downloads") -> None:
        self.download_dir = Path(download_dir).resolve()
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _build_ytdl_base_opts(self) -> dict[str, Any]:
        return {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "restrictfilenames": False,
        }

    def _extract_metadata_sync(
        self,
        url: str,
        custom_opts: dict[str, Any] | None,
    ) -> MediaMetadata:
        opts = self._build_ytdl_base_opts()
        if custom_opts:
            opts.update(custom_opts)

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise ValueError(f"No metadata returned for URL: {url}")

            return MediaMetadata(
                title=info.get("title") or "media",
                artist=info.get("artist") or info.get("creator") or info.get("uploader"),
                track=info.get("track") or info.get("title"),
                album=info.get("album"),
                duration=info.get("duration"),
                thumbnail=info.get("thumbnail"),
                width=info.get("width"),
                height=info.get("height"),
                filesize=info.get("filesize") or info.get("filesize_approx"),
                extractor=info.get("extractor") or "",
                formats=info.get("formats") or [],
            )

    async def extract_metadata(
        self,
        url: str,
        custom_opts: dict[str, Any] | None = None,
    ) -> MediaMetadata:
        """Extract media metadata asynchronously without downloading the stream."""
        return await asyncio.to_thread(self._extract_metadata_sync, url, custom_opts)

    def _download_sync(
        self,
        url: str,
        temp_dir: Path,
        custom_opts: dict[str, Any] | None,
    ) -> DownloadedMedia:
        opts = self._build_ytdl_base_opts()

        filename_template = "%(id)s.%(ext)s"
        merged_opts = dict(custom_opts) if custom_opts else {}
        if "outtmpl" in merged_opts:
            filename_template = merged_opts.pop("outtmpl")

        opts["outtmpl"] = str(temp_dir / filename_template)
        opts.update(merged_opts)

        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise ValueError(f"Download returned empty metadata for URL: {url}")

            expected_filepath = Path(ydl.prepare_filename(info))
            if expected_filepath.exists():
                file_path = expected_filepath
            else:
                candidates = [
                    f
                    for f in temp_dir.iterdir()
                    if f.is_file()
                    and not f.name.endswith(".temp")
                    and not f.name.endswith(".part")
                    and f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
                ]
                if candidates:
                    file_path = max(candidates, key=lambda f: f.stat().st_size)
                else:
                    raise FileNotFoundError(f"Downloaded media file not found in {temp_dir}")

            filesize = file_path.stat().st_size

            # Discover downloaded thumbnail cover art if present
            thumbnail_path: Path | None = None
            thumb_files = [
                f
                for f in temp_dir.iterdir()
                if f.is_file() and f != file_path and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if thumb_files:
                thumbnail_path = thumb_files[0]

            return DownloadedMedia(
                filepath=file_path,
                title=info.get("title") or "media",
                artist=info.get("artist") or info.get("creator") or info.get("uploader"),
                track=info.get("track") or info.get("title"),
                album=info.get("album"),
                duration=info.get("duration"),
                thumbnail_path=thumbnail_path,
                width=info.get("width"),
                height=info.get("height"),
                filesize=filesize,
                temp_dir=temp_dir,
            )

    async def download(
        self,
        url: str,
        custom_opts: dict[str, Any] | None = None,
    ) -> DownloadedMedia:
        """Download media into an isolated directory with automatic error cleanup."""
        job_id = uuid.uuid4().hex
        temp_dir = self.download_dir / f"job_{job_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            return await asyncio.to_thread(self._download_sync, url, temp_dir, custom_opts)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.exception("Failed to download media from %s, purged temp dir %s", url, temp_dir)
            raise
