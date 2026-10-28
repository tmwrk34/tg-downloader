"""
Platform adapter for Pinterest image, video, and idea pin (carousel) downloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from downloader.core import (
    _DEFAULT_USER_AGENT as _USER_AGENT,
)
from downloader.core import (
    _shortlink_cache as _shortlink_cache,
)
from downloader.core import (
    download_url_to_file,
    resolve_shortlink_url,
)
from downloader.models import DownloadedMedia
from downloader.platforms.base import BasePlatformService
from filters.patterns import MEDIA_PATTERNS, Platform

logger = logging.getLogger(__name__)

_PIN_ID_RE = re.compile(r"/pin/(?:[\w-]+--)?(\d+)")
_RELAY_PAYLOAD_RE = re.compile(
    r'window\.__PWS_RELAY_REGISTER_COMPLETED_REQUEST__\s*\(\s*"[^"]*"\s*,\s*(\{.*?\})\s*\);',
    re.DOTALL,
)


def resolve_pinterest_url(url: str, timeout: int = 10) -> str:
    """Resolve Pinterest shortlinks (pin.it) to their canonical URL with thread-safe LRU caching."""
    return resolve_shortlink_url(url, ("pin.it",), timeout=timeout)


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


def _extract_video_candidate(data: Any) -> dict[str, Any] | None:
    """Recursively search for highest-resolution direct MP4 video in Pinterest metadata."""
    candidates: list[dict[str, Any]] = []

    def recurse(node: Any) -> None:
        if not isinstance(node, (dict, list)):
            return
        if isinstance(node, dict):
            url = node.get("url")
            if (
                isinstance(url, str)
                and ".mp4" in url
                and "hls" not in url.lower()
                and not url.endswith(".m3u8")
            ):
                candidates.append(node)
            for v in node.values():
                recurse(v)
        elif isinstance(node, list):
            for item in node:
                recurse(item)

    recurse(data)
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda v: (v.get("width") or 0) * (v.get("height") or 0),
    )


def _pin_richness_score(pin: dict[str, Any]) -> int:
    """Score candidate pin payloads so the richest query result is preferred."""
    score = 0
    if (pin.get("carouselData") or {}).get("carouselSlots"):
        score += 200
    if pin.get("storyPinData"):
        score += 100
    if pin.get("videos"):
        score += 100
    if _extract_video_candidate(pin):
        score += 300
    score += len(pin)
    return score


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
        candidate_pins: list[dict[str, Any]] = []
        for json_payload in matches:
            try:
                parsed = json.loads(json_payload)
                data = parsed.get("data", {})
                if "v3GetPinQueryv2" in data:
                    pin = data["v3GetPinQueryv2"].get("data")
                    if pin and isinstance(pin, dict):
                        candidate_pins.append(pin)
            except Exception:
                continue

        if candidate_pins:
            return max(candidate_pins, key=_pin_richness_score)

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

            # 2. Check for Multi-Page Story / Idea Pin (multiple pages)
            story_pin_data = pin_data.get("storyPinData") or {}
            pages = story_pin_data.get("pages") or []

            if len(pages) > 1:
                gallery_items = []
                total_filesize = 0

                for idx, page in enumerate(pages):
                    video_cand = _extract_video_candidate(page)
                    if video_cand and video_cand.get("url"):
                        vid_path = temp_dir / f"page_{idx:02d}.mp4"
                        download_url_to_file(video_cand["url"], vid_path)
                        gallery_items.append(("video", vid_path))
                        total_filesize += vid_path.stat().st_size
                    else:
                        img_dict = page.get("image", {}).get("images", {})
                        img_url = (
                            _extract_slot_image_url(img_dict)
                            or _extract_slot_image_url(page)
                            or (page.get("images_orig") or {}).get("url")
                            or (page.get("images_1200x") or {}).get("url")
                            or (page.get("images_736x") or {}).get("url")
                        )
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

            # 3. Check for single Video Pin (in storyPinData, pin_data['videos'], or anywhere in pin_data)
            video_cand = None
            if pages:
                video_cand = _extract_video_candidate(pages[0])
            if not video_cand:
                video_cand = _extract_video_candidate(pin_data.get("videos") or {})
            if not video_cand:
                video_cand = _extract_video_candidate(pin_data)

            if video_cand and video_cand.get("url"):
                vid_path = temp_dir / f"{pin_id}.mp4"
                download_url_to_file(video_cand["url"], vid_path)
                dur = video_cand.get("duration")
                duration = (
                    int(round(dur / 1000)) if (dur and dur > 100) else (int(dur) if dur else None)
                )
                return DownloadedMedia(
                    filepath=vid_path,
                    title=title,
                    artist=uploader,
                    duration=duration,
                    width=video_cand.get("width"),
                    height=video_cand.get("height"),
                    filesize=vid_path.stat().st_size,
                    temp_dir=temp_dir,
                )

            # 4. Check for single Photo Pin
            orig_url = None
            if pages:
                orig_url = (
                    _extract_slot_image_url(pages[0])
                    or (pages[0].get("images_orig") or {}).get("url")
                    or (pages[0].get("images_1200x") or {}).get("url")
                    or (pages[0].get("images_736x") or {}).get("url")
                )
            if not orig_url:
                orig_url = (
                    _extract_slot_image_url(pin_data)
                    or (pin_data.get("images_orig") or {}).get("url")
                    or (pin_data.get("images_1200x") or {}).get("url")
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
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
