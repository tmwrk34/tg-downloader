"""
Platform adapter for Pinterest image, video, and idea pin (carousel) downloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from downloader.core import download_url_to_file
from downloader.models import DownloadedMedia
from downloader.platforms.base import BasePlatformService
from filters.patterns import MEDIA_PATTERNS, Platform

logger = logging.getLogger(__name__)

_SHORTLINK_CACHE_MAX = 512
_shortlink_cache: OrderedDict[str, str] = OrderedDict()
_shortlink_lock = threading.Lock()

_PIN_ID_RE = re.compile(r"/pin/(?:[\w-]+--)?(\d+)")
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def resolve_pinterest_url(url: str, timeout: int = 10) -> str:
    """Resolve Pinterest shortlinks (pin.it) to their canonical URL with thread-safe LRU caching."""
    if "pin.it" not in url:
        return url

    with _shortlink_lock:
        if url in _shortlink_cache:
            _shortlink_cache.move_to_end(url)
            return _shortlink_cache[url]

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _USER_AGENT},
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
        logger.warning("Failed to resolve Pinterest shortlink %s: %s", url, e)
        return url


def extract_pin_id(url: str) -> str | None:
    """Extract numeric pin ID from a Pinterest URL."""
    m = _PIN_ID_RE.search(url)
    return m.group(1) if m else None


class PinterestService(BasePlatformService):
    """Platform adapter for Pinterest images, videos, and multi-image idea pins."""

    @property
    def platform_name(self) -> str:
        return "pinterest"

    def is_supported(self, url: str) -> bool:
        pattern = MEDIA_PATTERNS.get(Platform.PINTEREST)
        return bool(pattern and pattern.search(url))

    def get_ytdl_options(self) -> dict[str, Any]:
        return {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "noplaylist": True,
            "ignore_no_formats_error": True,
        }

    async def download(
        self,
        url: str,
        extra_opts: dict[str, Any] | None = None,
    ) -> DownloadedMedia:
        """
        Download Pinterest media:
        1. Resolves shortlinks (pin.it) to canonical URLs.
        2. Attempts standard yt-dlp download (for video pins and basic photos).
        3. If yt-dlp fails or returns incomplete media, falls back to the native Pinterest Pin API.
        """
        resolved_url = await asyncio.to_thread(resolve_pinterest_url, url)

        # 1. Try yt-dlp first
        try:
            downloaded = await super().download(resolved_url, extra_opts=extra_opts)
            return downloaded
        except Exception as yt_err:
            logger.info(
                "yt-dlp extraction failed for Pinterest URL %s (%s). Falling back to direct API scraper...",
                resolved_url,
                yt_err,
            )

        # 2. Fallback to direct Pinterest unauth API scraper
        return await asyncio.to_thread(self._download_direct_pin_sync, resolved_url)

    def _fetch_pin_data_sync(self, pin_id: str, timeout: int = 15) -> dict[str, Any]:
        """Fetch unauthenticated pin metadata from Pinterest's public PinResource API."""
        api_data = {
            "options": {
                "field_set_key": "unauth_react_main_pin",
                "id": pin_id,
            },
            "context": {},
        }
        query_str = urllib.parse.urlencode({"data": json.dumps(api_data)})
        endpoint = f"https://www.pinterest.com/resource/PinResource/get/?{query_str}"

        req = urllib.request.Request(
            endpoint,
            headers={
                "User-Agent": _USER_AGENT,
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
            data = json.loads(content)
            return data.get("resource_response", {}).get("data", {})

    def _download_direct_pin_sync(self, url: str) -> DownloadedMedia:
        """Synchronously download a Pinterest pin (single image, video, or multi-page idea pin) via direct API."""
        pin_id = extract_pin_id(url)
        if not pin_id:
            raise ValueError(f"Could not extract Pinterest pin ID from URL: {url}")

        pin_data = self._fetch_pin_data_sync(pin_id)
        if not pin_data:
            raise ValueError(f"No pin data returned from Pinterest API for pin {pin_id}")

        temp_dir = self.core.download_dir / f"job_{uuid.uuid4().hex[:12]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            title = pin_data.get("title") or pin_data.get("grid_title") or f"Pinterest Pin {pin_id}"
            uploader = pin_data.get("closeup_attribution", {}).get("full_name") or "Pinterest"

            # Check for Story Pin / Idea Pin Carousel (multiple pages)
            story_pin_data = pin_data.get("story_pin_data") or {}
            pages = story_pin_data.get("pages") or []

            if len(pages) > 1:
                gallery_items: list[tuple[str, Path]] = []
                total_filesize = 0

                for idx, page in enumerate(pages):
                    # Check for video blocks
                    video_blocks = [b for b in page.get("blocks", []) if b.get("type") == "video"]
                    video_list = (
                        video_blocks[0].get("video", {}).get("video_list", {})
                        if video_blocks
                        else {}
                    )

                    best_video_url = None
                    if video_list:
                        # Prefer non-hls mp4 formats
                        mp4_candidates = [
                            v["url"]
                            for k, v in video_list.items()
                            if isinstance(v, dict)
                            and v.get("url")
                            and "hls" not in k.lower()
                            and not v["url"].endswith(".m3u8")
                        ]
                        if mp4_candidates:
                            best_video_url = mp4_candidates[0]

                    if best_video_url:
                        vid_path = temp_dir / f"page_{idx:02d}.mp4"
                        download_url_to_file(best_video_url, vid_path)
                        gallery_items.append(("video", vid_path))
                        total_filesize += vid_path.stat().st_size
                    else:
                        # Photo page
                        img_dict = page.get("image", {}).get("images", {})
                        orig = img_dict.get("originals") or img_dict.get("orig")
                        img_url = orig.get("url") if orig else None
                        if not img_url and img_dict:
                            # Pick largest available resolution
                            largest = max(
                                img_dict.values(),
                                key=lambda i: (i.get("width") or 0) * (i.get("height") or 0),
                            )
                            img_url = largest.get("url")

                        if img_url:
                            img_path = temp_dir / f"page_{idx:02d}.jpg"
                            download_url_to_file(img_url, img_path)
                            gallery_items.append(("photo", img_path))
                            total_filesize += img_path.stat().st_size

                if gallery_items:
                    return DownloadedMedia(
                        filepath=gallery_items[0][1],
                        title=title,
                        artist=uploader,
                        temp_dir=temp_dir,
                        filesize=total_filesize,
                        is_gallery=True,
                        gallery_items=gallery_items,
                    )

            # Check for single Video Pin
            videos = pin_data.get("videos", {}).get("video_list", {})
            if videos:
                mp4_candidates = [
                    v
                    for k, v in videos.items()
                    if isinstance(v, dict)
                    and v.get("url")
                    and "hls" not in k.lower()
                    and not v["url"].endswith(".m3u8")
                ]
                if mp4_candidates:
                    best_v = max(
                        mp4_candidates,
                        key=lambda v: (v.get("width") or 0) * (v.get("height") or 0),
                    )
                    vid_path = temp_dir / f"{pin_id}.mp4"
                    download_url_to_file(best_v["url"], vid_path)
                    return DownloadedMedia(
                        filepath=vid_path,
                        title=title,
                        artist=uploader,
                        duration=int(round(best_v.get("duration", 0) / 1000))
                        if best_v.get("duration")
                        else None,
                        width=best_v.get("width"),
                        height=best_v.get("height"),
                        filesize=vid_path.stat().st_size,
                        temp_dir=temp_dir,
                    )

            # Check for single Image Pin
            images = pin_data.get("images", {})
            orig = images.get("orig") or images.get("originals")
            best_img_url = orig.get("url") if orig else None

            if not best_img_url and images:
                largest = max(
                    images.values(),
                    key=lambda i: (
                        (i.get("width") or 0) * (i.get("height") or 0) if isinstance(i, dict) else 0
                    ),
                )
                if isinstance(largest, dict):
                    best_img_url = largest.get("url")

            if best_img_url:
                img_path = temp_dir / f"{pin_id}.jpg"
                download_url_to_file(best_img_url, img_path)
                return DownloadedMedia(
                    filepath=img_path,
                    title=title,
                    artist=uploader,
                    filesize=img_path.stat().st_size,
                    temp_dir=temp_dir,
                )

            raise ValueError(f"No downloadable media formats found for Pinterest pin {pin_id}")

        except Exception:
            # Clean up temporary directory on failure
            for f in temp_dir.iterdir():
                f.unlink(missing_ok=True)
            temp_dir.rmdir()
            raise
