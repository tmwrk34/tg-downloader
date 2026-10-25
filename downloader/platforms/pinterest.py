"""
Platform adapter for Pinterest image, video, and idea pin (carousel) downloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
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
_RELAY_PAYLOAD_RE = re.compile(
    r'window\.__PWS_RELAY_REGISTER_COMPLETED_REQUEST__\s*\(\s*"[^"]*"\s*,\s*(\{.*?\})\s*\);',
    re.DOTALL,
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
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
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


def _promote_to_original_url(image_url: str) -> str:
    """Promote resized pinimg URL (e.g. 1200x, 736x, 236x) to full uncompressed original."""
    promoted = re.sub(r"/[0-9]+x[0-9]*/", "/originals/", image_url)
    return re.sub(r"/[0-9]+x/", "/originals/", promoted)


def _extract_slot_image_url(slot: dict[str, Any]) -> str | None:
    """Extract highest-resolution image URL from a carousel slot dictionary."""
    orig = slot.get("images_orig") or slot.get("orig")
    if isinstance(orig, dict) and orig.get("url"):
        return orig["url"]

    for k in ("images_1200x", "images_736x", "images_564x", "images_474x", "images_236x"):
        val = slot.get(k)
        if isinstance(val, dict) and val.get("url"):
            return _promote_to_original_url(val["url"])

    for k, v in slot.items():
        if "images" in k and isinstance(v, dict) and v.get("url"):
            return _promote_to_original_url(v["url"])

    return None


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
        2. Scrapes Pinterest webpage HTML and parses native Relay data (carousels, collages, videos, photos).
        3. If HTML parsing fails or no formats found, falls back to yt-dlp.
        """
        resolved_url = await asyncio.to_thread(resolve_pinterest_url, url)

        # 1. Try native webpage HTML extraction first (handles multi-photo collages and full originals)
        try:
            return await asyncio.to_thread(self._download_direct_pin_sync, resolved_url)
        except Exception as direct_err:
            logger.info(
                "Direct Pinterest extraction failed for %s (%s). Falling back to yt-dlp...",
                resolved_url,
                direct_err,
            )

        # 2. Fallback to yt-dlp
        return await super().download(resolved_url, extra_opts=extra_opts)

    def _fetch_pin_data_sync(self, pin_id: str, timeout: int = 15) -> dict[str, Any]:
        """Fetch pin metadata by parsing embedded Relay request payloads from webpage HTML."""
        target_url = f"https://www.pinterest.com/pin/{pin_id}/"
        req = urllib.request.Request(
            target_url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8")

        matches = _RELAY_PAYLOAD_RE.findall(html)
        for json_payload in matches:
            try:
                parsed = json.loads(json_payload)
                data = parsed.get("data", {})
                if "v3GetPinQueryv2" in data:
                    pin = data["v3GetPinQueryv2"].get("data")
                    if pin:
                        return pin
            except Exception:
                continue

        # Secondary fallback: check __PWS_DATA__
        pws_match = re.search(r'<script[^>]*id="__PWS_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
        if pws_match:
            try:
                pws_data = json.loads(pws_match.group(1))
                props = pws_data.get("props", {}) or pws_data
                if "initialReduxState" in props:
                    pins = props["initialReduxState"].get("pins", {})
                    if pin_id in pins:
                        return pins[pin_id]
                    if pins:
                        return next(iter(pins.values()))
            except Exception:
                pass

        return {}

    def _download_direct_pin_sync(self, url: str) -> DownloadedMedia:
        """Synchronously download Pinterest media (multi-photo collage, video, or single photo)."""
        pin_id = extract_pin_id(url)
        if not pin_id:
            raise ValueError(f"Could not extract Pinterest pin ID from URL: {url}")

        pin_data = self._fetch_pin_data_sync(pin_id)
        if not pin_data:
            raise ValueError(f"No pin data returned from Pinterest HTML for pin {pin_id}")

        temp_dir = self.core.download_dir / f"job_{uuid.uuid4().hex[:12]}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            title = pin_data.get("title") or pin_data.get("grid_title") or f"Pinterest Pin {pin_id}"
            uploader = (
                pin_data.get("closeup_attribution", {}).get("full_name")
                or pin_data.get("pinner", {}).get("username")
                or "Pinterest"
            )

            # 1. Check for Carousel / Collage (multiple photo slots)
            carousel_data = pin_data.get("carouselData") or {}
            slots = carousel_data.get("carouselSlots") or []

            if len(slots) > 1:
                gallery_items: list[tuple[str, Path]] = []
                total_filesize = 0

                for idx, slot in enumerate(slots):
                    img_url = _extract_slot_image_url(slot)
                    if img_url:
                        img_path = temp_dir / f"slot_{idx:02d}.jpg"
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

            # 2. Check for Story Pin / Idea Pin (multi-page)
            story_pin_data = pin_data.get("storyPinData") or {}
            pages = story_pin_data.get("pages") or []

            if len(pages) > 1:
                gallery_items = []
                total_filesize = 0

                for idx, page in enumerate(pages):
                    video_blocks = [b for b in page.get("blocks", []) if b.get("type") == "video"]
                    video_list = (
                        video_blocks[0].get("video", {}).get("video_list", {})
                        if video_blocks
                        else {}
                    )

                    best_video_url = None
                    if video_list:
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
                        img_dict = page.get("image", {}).get("images", {})
                        img_url = _extract_slot_image_url(img_dict) or _extract_slot_image_url(page)
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

            # 3. Check for single Video Pin
            videos = pin_data.get("videos")
            if isinstance(videos, dict):
                video_list = videos.get("video_list") or {}
                mp4_candidates = [
                    v
                    for k, v in video_list.items()
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

            # 4. Check for single Photo Pin
            orig_url = (
                _extract_slot_image_url(pin_data)
                or (pin_data.get("images_orig") or {}).get("url")
                or (pin_data.get("images_736x") or {}).get("url")
            )

            if orig_url:
                orig_url = _promote_to_original_url(orig_url)
                img_path = temp_dir / f"{pin_id}.jpg"
                download_url_to_file(orig_url, img_path)
                return DownloadedMedia(
                    filepath=img_path,
                    title=title,
                    artist=uploader,
                    filesize=img_path.stat().st_size,
                    temp_dir=temp_dir,
                )

            raise ValueError(f"No downloadable media formats found for Pinterest pin {pin_id}")

        except Exception:
            for f in temp_dir.iterdir():
                f.unlink(missing_ok=True)
            temp_dir.rmdir()
            raise
