import asyncio
import logging
import re
import shutil
import threading
import urllib.request
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.extractor.tiktok import TikTokIE

from downloader.core import download_url_to_file
from downloader.models import DownloadedMedia, MediaMetadata
from downloader.platforms.base import BasePlatformService
from filters.patterns import MEDIA_PATTERNS, Platform

logger = logging.getLogger(__name__)

_SHORTLINK_CACHE_MAX = 512
_shortlink_cache: OrderedDict[str, str] = OrderedDict()
_shortlink_lock = threading.Lock()


def resolve_tiktok_url(url: str, timeout: int = 10) -> str:
    """Resolve TikTok shortlinks (vt.tiktok.com, vm.tiktok.com) to their canonical URL with LRU caching."""
    if "vt.tiktok.com" not in url and "vm.tiktok.com" not in url:
        return url

    with _shortlink_lock:
        if url in _shortlink_cache:
            _shortlink_cache.move_to_end(url)
            return _shortlink_cache[url]

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resolved = resp.geturl()
        if resolved and resolved != url:
            with _shortlink_lock:
                _shortlink_cache[url] = resolved
                if len(_shortlink_cache) > _SHORTLINK_CACHE_MAX:
                    _shortlink_cache.popitem(last=False)
        return resolved
    except Exception as e:
        logger.warning("Failed to resolve TikTok shortlink %s: %s", url, e)
        return url


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

    def _extract_photo_metadata_sync(self, url: str) -> MediaMetadata:
        norm_url = re.sub(r"/photo/(\d+)", r"/video/\1", url)
        m = re.search(r"/video/(\d+)", norm_url)
        video_id = m.group(1) if m else "media"

        ydl_opts = self.get_ytdl_options()
        ydl_opts.update(self.core._build_ytdl_base_opts())
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ie = TikTokIE(ydl)
            video_data, _ = ie._extract_web_data_and_status(norm_url, video_id)
            if not video_data:
                raise ValueError(f"Could not extract TikTok data for {url}")

            image_post = video_data.get("imagePost") or {}
            images = image_post.get("images", [])
            title = video_data.get("desc") or "TikTok Photos"
            author_info = video_data.get("author") or {}
            artist = author_info.get("nickname") or author_info.get("uniqueId")

            return MediaMetadata(
                title=title,
                artist=artist,
                is_gallery=bool(images),
                gallery_count=len(images),
                extractor="tiktok",
            )

    def _download_photo_slideshow_sync(self, url: str) -> DownloadedMedia:
        job_id = uuid.uuid4().hex
        temp_dir = self.core.download_dir / f"job_{job_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            norm_url = re.sub(r"/photo/(\d+)", r"/video/\1", url)
            m = re.search(r"/video/(\d+)", norm_url)
            video_id = m.group(1) if m else "media"

            ydl_opts = self.get_ytdl_options()
            ydl_opts.update(self.core._build_ytdl_base_opts())
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ie = TikTokIE(ydl)
                video_data, _ = ie._extract_web_data_and_status(norm_url, video_id)
                if not video_data:
                    raise ValueError(f"Could not extract TikTok data for {url}")

                image_post = video_data.get("imagePost") or {}
                images = image_post.get("images", [])
                if not images:
                    return self.core._download_sync(norm_url, temp_dir, ydl_opts)

                gallery_items: list[tuple[str, Path]] = []
                total_size = 0
                for idx, img in enumerate(images):
                    url_list = img.get("imageURL", {}).get("urlList", [])
                    if url_list:
                        target_path = temp_dir / f"slide_{idx:02d}.jpg"
                        download_url_to_file(url_list[0], target_path)
                        sz = target_path.stat().st_size
                        total_size += sz
                        gallery_items.append(("photo", target_path))

                music_info = video_data.get("music") or {}
                audio_url = music_info.get("playUrl")
                audio_path: Path | None = None
                if audio_url:
                    audio_path = temp_dir / "audio.mp3"
                    try:
                        download_url_to_file(audio_url, audio_path)
                        total_size += audio_path.stat().st_size
                    except Exception:
                        audio_path = None

                title = video_data.get("desc") or "TikTok Photos"
                author_info = video_data.get("author") or {}
                artist = author_info.get("nickname") or author_info.get("uniqueId")

                return DownloadedMedia(
                    filepath=gallery_items[0][1] if gallery_items else temp_dir,
                    title=title,
                    artist=artist,
                    filesize=total_size,
                    temp_dir=temp_dir,
                    is_gallery=True,
                    gallery_items=gallery_items,
                    extra_audio_path=audio_path,
                )
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    async def extract_metadata(
        self,
        url: str,
        extra_opts: dict[str, Any] | None = None,
    ) -> MediaMetadata:
        resolved_url = await asyncio.to_thread(resolve_tiktok_url, url)
        if "/photo/" in resolved_url:
            return await asyncio.to_thread(self._extract_photo_metadata_sync, resolved_url)
        norm_url = re.sub(r"/photo/(\d+)", r"/video/\1", resolved_url)
        return await super().extract_metadata(norm_url, extra_opts=extra_opts)

    async def download(
        self,
        url: str,
        extra_opts: dict[str, Any] | None = None,
    ) -> DownloadedMedia:
        resolved_url = await asyncio.to_thread(resolve_tiktok_url, url)
        if "/photo/" in resolved_url:
            return await asyncio.to_thread(self._download_photo_slideshow_sync, resolved_url)
        norm_url = re.sub(r"/photo/(\d+)", r"/video/\1", resolved_url)
        return await super().download(norm_url, extra_opts=extra_opts)
