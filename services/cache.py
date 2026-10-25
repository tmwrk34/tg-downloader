"""
Media cache service mapping normalized URLs to Telegram file_ids.
Provides high-performance L1 RAM LRU + L2 SQLite WAL storage to eliminate redundant downloads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from downloader.platforms.pinterest import resolve_pinterest_url
from downloader.platforms.tiktok import resolve_tiktok_url

logger = logging.getLogger(__name__)

# Pre-compiled platform normalization patterns
_IG_NORM_RE = re.compile(
    r"(?:instagram\.com|instagr\.am)/(?:reel|reels|p|tv|share/reel|share/p)/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
_YT_SHORT_NORM_RE = re.compile(
    r"youtube\.com/shorts/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
_YT_VIDEO_NORM_RE = re.compile(
    r"(?:youtube\.com/watch\?[^\s<>\"']*v=|youtu\.be/|youtube\.com/embed/|youtube\.com/live/)([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)
_TT_VIDEO_NORM_RE = re.compile(
    r"tiktok\.com/@[\w.-]+/video/(\d+)",
    re.IGNORECASE,
)
_TT_PHOTO_NORM_RE = re.compile(
    r"tiktok\.com/@[\w.-]+/photo/(\d+)",
    re.IGNORECASE,
)
_TT_SHORTLINK_NORM_RE = re.compile(
    r"(?:vm|vt)\.tiktok\.com/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
_PIN_NORM_RE = re.compile(
    r"pinterest\.[a-z.]+/pin/(?:[\w-]+--)?(\d+)",
    re.IGNORECASE,
)
_PIN_SHORTLINK_NORM_RE = re.compile(
    r"pin\.it/([a-zA-Z0-9]+)",
    re.IGNORECASE,
)
_TW_NORM_RE = re.compile(
    r"(?:twitter|x)\.com/[^/]+/status/(\d+)",
    re.IGNORECASE,
)


def resolve_default_cache_db() -> str:
    """Resolve the default SQLite path, falling back to local working directory if data/ is not writable."""
    env_path = os.getenv("MEDIA_CACHE_DB")
    if env_path:
        return env_path
    data_dir = Path("data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        return "data/media_cache.db"
    except (PermissionError, OSError):
        return "media_cache.db"


# Common tracking / referrer query parameters to strip
TRACKING_QUERY_PARAMS = {
    "igsh",
    "rpxt",
    "si",
    "feature",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "ref_src",
    "t",
    "s",
    "is_from_webapp",
    "sender_device",
    "share_id",
    "source",
}


@dataclass(frozen=True)
class CachedMedia:
    """Represents a cached media object in Telegram."""

    cache_key: str
    media_type: str  # "video", "photo", "audio", "album"
    file_id: str | None
    caption: str
    extra_data: dict[str, Any]
    created_at: float


def normalize_media_url(
    url: str,
    platform: str | None = None,
    quality: str | None = None,
) -> str:
    """
    Produce a canonical, deduplicated cache key for a media URL.
    Strips platform-specific tracking parameters, canonicalizes IDs,
    and appends optional format/quality markers.
    """
    clean_url_str = url.strip()

    # 1. Instagram normalization: /reel/ID, /p/ID, /tv/ID, /share/reel/ID
    ig_match = _IG_NORM_RE.search(clean_url_str)
    if ig_match:
        return f"instagram:{ig_match.group(1)}"

    # 2. YouTube normalization
    yt_short_match = _YT_SHORT_NORM_RE.search(clean_url_str)
    if yt_short_match:
        return f"youtube:short:{yt_short_match.group(1)}"

    yt_video_match = _YT_VIDEO_NORM_RE.search(clean_url_str)
    if yt_video_match:
        vid_id = yt_video_match.group(1)
        if quality:
            return f"youtube:{vid_id}:{quality}"
        return f"youtube:{vid_id}"

    # 3. TikTok normalization (for direct @user/video/ID or @user/photo/ID)
    tt_video_match = _TT_VIDEO_NORM_RE.search(clean_url_str)
    if tt_video_match:
        return f"tiktok:video:{tt_video_match.group(1)}"

    tt_photo_match = _TT_PHOTO_NORM_RE.search(clean_url_str)
    if tt_photo_match:
        return f"tiktok:photo:{tt_photo_match.group(1)}"

    tt_shortlink_match = _TT_SHORTLINK_NORM_RE.search(clean_url_str)
    if tt_shortlink_match:
        return f"tiktok:shortlink:{tt_shortlink_match.group(1)}"

    # 4. Pinterest normalization
    pin_match = _PIN_NORM_RE.search(clean_url_str)
    if pin_match:
        return f"pinterest:pin:{pin_match.group(1)}"

    pin_short_match = _PIN_SHORTLINK_NORM_RE.search(clean_url_str)
    if pin_short_match:
        return f"pinterest:shortlink:{pin_short_match.group(1)}"

    # 5. Twitter / X normalization
    tw_match = _TW_NORM_RE.search(clean_url_str)
    if tw_match:
        return f"twitter:status:{tw_match.group(1)}"

    # 6. Generic URL normalization (strip tracking parameters)
    parsed = urlparse(clean_url_str)
    query_dict = parse_qs(parsed.query, keep_blank_values=False)
    filtered_query = {k: v for k, v in query_dict.items() if k.lower() not in TRACKING_QUERY_PARAMS}

    new_query = urlencode(filtered_query, doseq=True)
    clean_path = parsed.path.rstrip("/") or "/"

    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    normalized = urlunparse(
        (
            parsed.scheme.lower() or "https",
            netloc,
            clean_path,
            "",
            new_query,
            "",
        )
    )

    prefix = f"{platform}:" if platform else "generic:"
    return f"{prefix}{normalized}"


async def get_canonical_cache_key(
    url: str,
    platform: str | None = None,
    quality: str | None = None,
) -> str:
    """
    Resolve potential redirects (such as TikTok shortlinks) asynchronously
    and return the canonical cache key.
    """
    clean_url = url.strip()
    if "vt.tiktok.com" in clean_url or "vm.tiktok.com" in clean_url:
        try:
            resolved = await asyncio.to_thread(resolve_tiktok_url, clean_url)
            return normalize_media_url(resolved, platform="tiktok", quality=quality)
        except Exception as e:
            logger.warning("Could not resolve TikTok shortlink %s for cache key: %s", clean_url, e)

    if "pin.it" in clean_url:
        try:
            resolved = await asyncio.to_thread(resolve_pinterest_url, clean_url)
            return normalize_media_url(resolved, platform="pinterest", quality=quality)
        except Exception as e:
            logger.warning(
                "Could not resolve Pinterest shortlink %s for cache key: %s", clean_url, e
            )

    return normalize_media_url(clean_url, platform=platform, quality=quality)


class MediaCache:
    """
    Thread-safe two-tier (L1 RAM LRU + L2 SQLite WAL) persistent media cache
    mapping normalized URLs to Telegram file_ids.
    """

    def __init__(self, db_path: str | Path | None = None, l1_max_size: int = 512) -> None:
        self.db_path = str(db_path) if db_path is not None else resolve_default_cache_db()
        self._is_memory = self.db_path == ":memory:"
        self._l1_max_size = l1_max_size
        self._l1_cache: OrderedDict[str, CachedMedia] = OrderedDict()
        self._lock = threading.Lock()
        self._conn = self._create_connection()
        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        if not self._is_memory and not self.db_path.startswith("file:"):
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.startswith("file:"):
            return sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """Create database tables and set optimal PRAGMA settings."""
        with self._lock:
            if not self._is_memory:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
                self._conn.execute("PRAGMA temp_store=MEMORY;")
                self._conn.execute("PRAGMA cache_size=-4000;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS media_cache (
                    cache_key TEXT PRIMARY KEY,
                    media_type TEXT NOT NULL,
                    file_id TEXT,
                    caption TEXT,
                    extra_data TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_cache_created ON media_cache(created_at);"
            )
            self._conn.commit()

    def _l1_put(self, entry: CachedMedia) -> None:
        self._l1_cache[entry.cache_key] = entry
        self._l1_cache.move_to_end(entry.cache_key)
        if len(self._l1_cache) > self._l1_max_size:
            self._l1_cache.popitem(last=False)

    def _sync_get(self, cache_key: str) -> CachedMedia | None:
        with self._lock:
            if cache_key in self._l1_cache:
                self._l1_cache.move_to_end(cache_key)
                return self._l1_cache[cache_key]

            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT cache_key, media_type, file_id, caption, extra_data, created_at
                FROM media_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            k, m_type, f_id, cap, extra_json, created_at = row
            try:
                extra_data = json.loads(extra_json) if extra_json else {}
            except Exception:
                extra_data = {}

            entry = CachedMedia(
                cache_key=k,
                media_type=m_type,
                file_id=f_id,
                caption=cap or "",
                extra_data=extra_data,
                created_at=created_at,
            )
            self._l1_put(entry)
            return entry

    def _sync_set(
        self,
        cache_key: str,
        media_type: str,
        file_id: str | None,
        caption: str,
        extra_data: dict[str, Any],
    ) -> None:
        with self._lock:
            extra_json = json.dumps(extra_data)
            now = time.time()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO media_cache (
                    cache_key, media_type, file_id, caption, extra_data, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, media_type, file_id, caption, extra_json, now),
            )
            self._conn.commit()
            self._l1_put(
                CachedMedia(
                    cache_key=cache_key,
                    media_type=media_type,
                    file_id=file_id,
                    caption=caption,
                    extra_data=dict(extra_data),
                    created_at=now,
                )
            )

    def _sync_delete(self, cache_key: str) -> bool:
        with self._lock:
            self._l1_cache.pop(cache_key, None)
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM media_cache WHERE cache_key = ?", (cache_key,))
            self._conn.commit()
            return cursor.rowcount > 0

    def _sync_count(self) -> int:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM media_cache")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def _sync_clear_expired(self, max_age_seconds: float) -> int:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM media_cache WHERE created_at < ?", (cutoff,))
            self._conn.commit()
            if cursor.rowcount > 0:
                expired_l1 = [k for k, v in self._l1_cache.items() if v.created_at < cutoff]
                for k in expired_l1:
                    self._l1_cache.pop(k, None)
            return cursor.rowcount

    async def get(self, cache_key: str) -> CachedMedia | None:
        """Retrieve cached media metadata by canonical cache key (fast-path L1 RAM check)."""
        with self._lock:
            if cache_key in self._l1_cache:
                self._l1_cache.move_to_end(cache_key)
                return self._l1_cache[cache_key]
        return await asyncio.to_thread(self._sync_get, cache_key)

    async def set(
        self,
        cache_key: str,
        media_type: str,
        file_id: str | None = None,
        caption: str = "",
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        """Store or update a cached media entry."""
        data = extra_data or {}
        await asyncio.to_thread(
            self._sync_set,
            cache_key,
            media_type,
            file_id,
            caption,
            data,
        )

    async def delete(self, cache_key: str) -> bool:
        """Delete a cache entry (e.g. after invalid file_id error)."""
        return await asyncio.to_thread(self._sync_delete, cache_key)

    async def count(self) -> int:
        """Return total number of cached entries."""
        return await asyncio.to_thread(self._sync_count)

    async def clear_expired(self, max_age_days: int = 30) -> int:
        """Purge entries older than the specified retention window."""
        max_age_seconds = max_age_days * 86400.0
        return await asyncio.to_thread(self._sync_clear_expired, max_age_seconds)

    def close(self) -> None:
        """Close the underlying SQLite connection and clear L1 cache."""
        with self._lock:
            self._l1_cache.clear()
            with contextlib.suppress(Exception):
                self._conn.close()
