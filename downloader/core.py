import asyncio
import logging
import os
import re
import shutil
import subprocess
import threading
import urllib.request
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yt_dlp

from downloader.models import DownloadedMedia, MediaMetadata

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_SHORTLINK_CACHE_MAX = 512
_shortlink_cache: OrderedDict[str, str] = OrderedDict()
_shortlink_lock = threading.Lock()


def resolve_shortlink_url(
    url: str,
    short_domains: tuple[str, ...],
    timeout: int = 10,
) -> str:
    """Resolve platform shortlinks to their canonical URL with thread-safe LRU caching."""
    if not any(domain in url for domain in short_domains):
        return url

    with _shortlink_lock:
        if url in _shortlink_cache:
            _shortlink_cache.move_to_end(url)
            return _shortlink_cache[url]

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _DEFAULT_USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
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
        logger.warning("Failed to resolve shortlink %s: %s", url, e)
        return url


def download_url_to_file(url: str, dest_path: Path, timeout: int = 30) -> None:
    """Download a remote media URL directly to disk with a 64KB streaming buffer."""
    headers = {
        "User-Agent": _DEFAULT_USER_AGENT,
        "Accept": "*/*",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file, length=65536)


_last_written_raw_cookies: str | None = None


def clean_artist_tag(
    artist: str | None,
    artists: list[str] | None = None,
    uploader: str | None = None,
) -> str | None:
    """Clean and deduplicate artist strings, stripping auto-generated YouTube channel artifacts."""
    if artists:
        deduped = list(dict.fromkeys([a.strip() for a in artists if a and a.strip()]))
        if deduped:
            return ", ".join(deduped)
    if artist:
        parts = [p.strip() for p in artist.split(",") if p.strip()]
        deduped = list(dict.fromkeys(parts))
        if deduped:
            return ", ".join(deduped)
    if uploader:
        u = uploader.strip()
        if u.endswith(" - Topic"):
            u = u[:-8].strip()
        return u
    return None


def sanitize_filename(name: str, max_bytes: int = 180) -> str:
    """
    Sanitize filename for filesystem compatibility and truncate so that
    its UTF-8 encoded length strictly stays within max_bytes (preserving extension).
    Prevents OSError [Errno 36] File name too long on Linux filesystems (NAME_MAX=255).
    """
    clean_name = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", name).strip()
    clean_name = re.sub(r"\s+", " ", clean_name)
    clean_name = clean_name.rstrip(". ")
    if not clean_name:
        return "audio"

    path = Path(clean_name)
    suffix = path.suffix.lower()
    stem = path.stem or "audio"

    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= max_bytes:
        suffix = suffix[:10]
        suffix_bytes = suffix.encode("utf-8")

    max_stem_bytes = max(10, max_bytes - len(suffix_bytes))
    stem_bytes = stem.encode("utf-8")

    if len(stem_bytes) > max_stem_bytes:
        truncated_stem = stem_bytes[:max_stem_bytes].decode("utf-8", errors="ignore").rstrip(". ")
        if not truncated_stem:
            truncated_stem = "audio"
        return f"{truncated_stem}{suffix}"

    return f"{stem}{suffix}"


class MediaDownloaderCore:
    """Core asynchronous media downloader using yt-dlp."""

    def __init__(self, download_dir: Path | str = "./downloads") -> None:
        self.download_dir = Path(download_dir).resolve()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_jobs()

    def _cleanup_stale_jobs(self) -> None:
        """Purge any orphaned job_* directories left over from previous interrupted runs."""
        try:
            for item in self.download_dir.glob("job_*"):
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    logger.info("Purged stale temporary job directory on startup: %s", item)
        except Exception as e:
            logger.debug("Could not sweep stale job directories in %s: %s", self.download_dir, e)

    def _get_cookie_file(self) -> str | None:
        """Locate an existing cookie file or initialize one from environment variables."""
        global _last_written_raw_cookies

        env_file = os.getenv("COOKIES_FILE")
        if env_file and Path(env_file).is_file():
            return str(Path(env_file).resolve())

        raw_cookies = os.getenv("INSTAGRAM_COOKIES") or os.getenv("COOKIES")
        if raw_cookies and raw_cookies.strip():
            stripped = raw_cookies.strip()
            tmp_path = Path("/tmp/cookies.txt")
            try:
                if _last_written_raw_cookies != stripped or not tmp_path.is_file():
                    tmp_path.write_text(stripped, encoding="utf-8")
                    _last_written_raw_cookies = stripped
                return str(tmp_path)
            except Exception as e:
                logger.warning("Could not write cookies to /tmp/cookies.txt: %s", e)

        candidates = [
            Path("/app/data/cookies.txt"),
            Path("./data/cookies.txt"),
            Path("./cookies.txt"),
            Path("/tmp/cookies.txt"),
        ]
        for c in candidates:
            if c.is_file():
                return str(c.resolve())

        return None

    def _build_ytdl_base_opts(self) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "restrictfilenames": False,
            "ignore_no_formats_error": True,
            "socket_timeout": 30,
            "concurrent_fragment_downloads": 2,
            "postprocessor_args": {"ffmpeg": ["-threads", "2"]},
        }
        cookie_file = self._get_cookie_file()
        if cookie_file:
            opts["cookiefile"] = cookie_file
        return opts

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

            entries = list(info.get("entries") or [])
            is_gallery = bool(entries)
            gallery_count = len(entries) if is_gallery else 0

            # Retain only the lightweight fields needed for size estimation to prevent RAM bloat
            slim_formats = [
                {
                    "height": f.get("height"),
                    "vcodec": f.get("vcodec"),
                    "acodec": f.get("acodec"),
                    "filesize": f.get("filesize"),
                    "filesize_approx": f.get("filesize_approx"),
                }
                for f in (info.get("formats") or [])
                if isinstance(f, dict)
            ]

            return MediaMetadata(
                title=info.get("title") or "media",
                artist=clean_artist_tag(
                    artist=info.get("artist"),
                    artists=info.get("artists"),
                    uploader=info.get("uploader") or info.get("creator"),
                ),
                track=info.get("track") or info.get("title"),
                album=info.get("album"),
                duration=int(round(info["duration"])) if info.get("duration") is not None else None,
                thumbnail=info.get("thumbnail"),
                width=int(round(info["width"])) if info.get("width") is not None else None,
                height=int(round(info["height"])) if info.get("height") is not None else None,
                filesize=info.get("filesize") or info.get("filesize_approx"),
                extractor=info.get("extractor") or "",
                formats=slim_formats,
                is_gallery=is_gallery,
                gallery_count=gallery_count,
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
            info = ydl.extract_info(url, download=False)
            if not info:
                raise ValueError(f"Download returned empty metadata for URL: {url}")

            clean_artist = clean_artist_tag(
                artist=info.get("artist"),
                artists=info.get("artists"),
                uploader=info.get("uploader") or info.get("creator"),
            )
            clean_track = info.get("track") or info.get("title")
            entries = list(info.get("entries") or [])

            # Case 1: Multi-item gallery or carousel (e.g. Instagram carousel)
            if entries:
                gallery_items: list[tuple[str, Path]] = []
                total_filesize = 0
                for idx, entry in enumerate(entries):
                    v_formats = [f for f in entry.get("formats", []) if f.get("vcodec") != "none"]
                    if v_formats:
                        best_v = max(
                            v_formats, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0)
                        )
                        item_path = temp_dir / f"item_{idx:02d}_{entry.get('id', 'media')}.mp4"
                        download_url_to_file(best_v["url"], item_path)
                        total_filesize += item_path.stat().st_size
                        gallery_items.append(("video", item_path))
                    elif entry.get("thumbnails"):
                        best_t = max(
                            entry["thumbnails"],
                            key=lambda t: (
                                t.get("preference") or 0,
                                (t.get("width") or 0) * (t.get("height") or 0),
                            ),
                        )
                        item_path = temp_dir / f"item_{idx:02d}_{entry.get('id', 'media')}.jpg"
                        download_url_to_file(best_t["url"], item_path)
                        total_filesize += item_path.stat().st_size
                        gallery_items.append(("photo", item_path))

                if gallery_items:
                    return DownloadedMedia(
                        filepath=gallery_items[0][1],
                        title=info.get("title") or "Gallery",
                        artist=clean_artist,
                        track=clean_track,
                        album=info.get("album"),
                        duration=int(round(info["duration"]))
                        if info.get("duration") is not None
                        else None,
                        filesize=total_filesize,
                        temp_dir=temp_dir,
                        is_gallery=True,
                        gallery_items=gallery_items,
                    )

            # Case 2: Single photo post without video or audio formats
            v_formats = [f for f in info.get("formats", []) if f.get("vcodec") != "none"]
            a_formats = [f for f in info.get("formats", []) if f.get("acodec") != "none"]
            if not v_formats and not a_formats and info.get("thumbnails"):
                best_t = max(
                    info["thumbnails"],
                    key=lambda t: (
                        t.get("preference") or 0,
                        (t.get("width") or 0) * (t.get("height") or 0),
                    ),
                )
                photo_path = temp_dir / f"{info.get('id', 'photo')}.jpg"
                download_url_to_file(best_t["url"], photo_path)
                return DownloadedMedia(
                    filepath=photo_path,
                    title=info.get("title") or "photo",
                    artist=clean_artist,
                    track=clean_track,
                    album=info.get("album"),
                    duration=int(round(info["duration"]))
                    if info.get("duration") is not None
                    else None,
                    filesize=photo_path.stat().st_size,
                    temp_dir=temp_dir,
                )

            # Case 3: Standard single video or audio stream
            ydl.process_info(info)

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

            # For audio downloads, rename output file to clean 'Artist - Track.ext'
            if file_path.suffix.lower() in {".mp3", ".m4a", ".flac", ".ogg", ".opus"}:
                clean_title_part = clean_track or info.get("title") or "Audio"
                if clean_artist:
                    new_filename = f"{clean_artist} - {clean_title_part}{file_path.suffix.lower()}"
                else:
                    new_filename = f"{clean_title_part}{file_path.suffix.lower()}"
                safe_name = sanitize_filename(new_filename, max_bytes=180)
                renamed_path = temp_dir / safe_name
                try:
                    if renamed_path != file_path and not renamed_path.exists():
                        file_path.rename(renamed_path)
                        file_path = renamed_path
                except OSError as rename_err:
                    logger.warning("Could not rename audio file to '%s': %s", safe_name, rename_err)

            # Discover downloaded thumbnail cover art if present (.jpg, .jpeg, .png)
            thumbnail_path: Path | None = None
            thumb_files = [
                f
                for f in temp_dir.iterdir()
                if f.is_file() and f != file_path and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if thumb_files:
                thumbnail_path = thumb_files[0]
            else:
                # If only .webp thumbnail exists, convert it to .jpg using ffmpeg
                webp_files = [
                    f
                    for f in temp_dir.iterdir()
                    if f.is_file() and f != file_path and f.suffix.lower() == ".webp"
                ]
                if webp_files:
                    webp_file = webp_files[0]
                    jpg_target = webp_file.with_suffix(".jpg")
                    try:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-threads",
                                "2",
                                "-i",
                                str(webp_file),
                                str(jpg_target),
                            ],
                            check=False,
                            timeout=30,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        if jpg_target.exists():
                            thumbnail_path = jpg_target
                    except Exception:
                        logger.warning("Failed to convert webp thumbnail to jpg for %s", file_path)

            return DownloadedMedia(
                filepath=file_path,
                title=info.get("title") or "media",
                artist=clean_artist,
                track=clean_track,
                album=info.get("album"),
                duration=int(round(info["duration"])) if info.get("duration") is not None else None,
                thumbnail_path=thumbnail_path,
                width=int(round(info["width"])) if info.get("width") is not None else None,
                height=int(round(info["height"])) if info.get("height") is not None else None,
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
