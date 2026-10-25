"""
Media cache service mapping normalized URLs to Telegram file_ids.
Provides high-performance SQLite-backed storage to eliminate redundant downloads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from downloader.platforms.tiktok import resolve_tiktok_url

logger = logging.getLogger(__name__)


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
    ig_match = re.search(
        r"(?:instagram\.com|instagr\.am)/(?:reel|reels|p|tv|share/reel|share/p)/([a-zA-Z0-9_-]+)",
        clean_url_str,
        re.IGNORECASE,
    )
    if ig_match:
        shortcode = ig_match.group(1)
        return f"instagram:{shortcode}"

    # 2. YouTube normalization
    # Shorts
    yt_short_match = re.search(
        r"youtube\.com/shorts/([a-zA-Z0-9_-]+)",
        clean_url_str,
        re.IGNORECASE,
    )
    if yt_short_match:
        return f"youtube:short:{yt_short_match.group(1)}"

    # Standard YouTube video
    yt_video_match = re.search(
        r"(?:youtube\.com/watch\?[^\s<>\"']*v=|youtu\.be/|youtube\.com/embed/|youtube\.com/live/)([a-zA-Z0-9_-]{11})",
        clean_url_str,
        re.IGNORECASE,
    )
    if yt_video_match:
        vid_id = yt_video_match.group(1)
        if quality:
            return f"youtube:{vid_id}:{quality}"
        return f"youtube:{vid_id}"

    # 3. TikTok normalization (for direct @user/video/ID or @user/photo/ID)
    tt_video_match = re.search(
        r"tiktok\.com/@[\w.-]+/video/(\d+)",
        clean_url_str,
        re.IGNORECASE,
    )
    if tt_video_match:
        return f"tiktok:video:{tt_video_match.group(1)}"

    tt_photo_match = re.search(
        r"tiktok\.com/@[\w.-]+/photo/(\d+)",
        clean_url_str,
        re.IGNORECASE,
    )
    if tt_photo_match:
        return f"tiktok:photo:{tt_photo_match.group(1)}"

    # TikTok shortlink without redirect resolution fallback
    tt_shortlink_match = re.search(
        r"(?:vm|vt)\.tiktok\.com/([a-zA-Z0-9_-]+)",
        clean_url_str,
        re.IGNORECASE,
    )
    if tt_shortlink_match:
        return f"tiktok:shortlink:{tt_shortlink_match.group(1)}"

    # 4. Twitter / X normalization
    tw_match = re.search(
        r"(?:twitter|x)\.com/[^/]+/status/(\d+)",
        clean_url_str,
        re.IGNORECASE,
    )
    if tw_match:
        return f"twitter:status:{tw_match.group(1)}"

    # 5. Generic URL normalization (strip tracking parameters)
    parsed = urlparse(clean_url_str)
    query_dict = parse_qs(parsed.query, keep_blank_values=False)
    filtered_query = {k: v for k, v in query_dict.items() if k.lower() not in TRACKING_QUERY_PARAMS}

    # Reassemble stripped URL
    new_query = urlencode(filtered_query, doseq=True)
    clean_path = parsed.path.rstrip("/")
    if not clean_path:
        clean_path = "/"

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

    return normalize_media_url(clean_url, platform=platform, quality=quality)


class MediaCache:
    """
    Thread-safe SQLite persistent media cache mapping normalized URLs to Telegram file_ids.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path) if db_path is not None else resolve_default_cache_db()
        self._is_memory = self.db_path == ":memory:"
        self._mem_conn: sqlite3.Connection | None = (
            sqlite3.connect(":memory:", check_same_thread=False) if self._is_memory else None
        )
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory and self._mem_conn is not None:
            return self._mem_conn
        if self.db_path.startswith("file:"):
            return sqlite3.connect(self.db_path, uri=True)
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create database tables and set optimal PRAGMA settings."""
        if not self._is_memory and not self.db_path.startswith("file:"):
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        try:
            if not self._is_memory:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_cache_created ON media_cache(created_at);"
            )
            conn.commit()
        finally:
            if not self._is_memory:
                conn.close()

    def _sync_get(self, cache_key: str) -> CachedMedia | None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
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

            return CachedMedia(
                cache_key=k,
                media_type=m_type,
                file_id=f_id,
                caption=cap or "",
                extra_data=extra_data,
                created_at=created_at,
            )
        finally:
            if not self._is_memory:
                conn.close()

    def _sync_set(
        self,
        cache_key: str,
        media_type: str,
        file_id: str | None,
        caption: str,
        extra_data: dict[str, Any],
    ) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            extra_json = json.dumps(extra_data)
            now = time.time()
            cursor.execute(
                """
                INSERT OR REPLACE INTO media_cache (
                    cache_key, media_type, file_id, caption, extra_data, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cache_key, media_type, file_id, caption, extra_json, now),
            )
            conn.commit()
        finally:
            if not self._is_memory:
                conn.close()

    def _sync_delete(self, cache_key: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM media_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            if not self._is_memory:
                conn.close()

    def _sync_count(self) -> int:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM media_cache")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            if not self._is_memory:
                conn.close()

    def _sync_clear_expired(self, max_age_seconds: float) -> int:
        cutoff = time.time() - max_age_seconds
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM media_cache WHERE created_at < ?", (cutoff,))
            conn.commit()
            return cursor.rowcount
        finally:
            if not self._is_memory:
                conn.close()

    async def get(self, cache_key: str) -> CachedMedia | None:
        """Retrieve cached media metadata by canonical cache key."""
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
