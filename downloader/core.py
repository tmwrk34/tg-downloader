import asyncio
import logging
import re
import shutil
import subprocess
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import yt_dlp

from downloader.models import DownloadedMedia, MediaMetadata

logger = logging.getLogger(__name__)


def download_url_to_file(url: str, dest_path: Path, timeout: int = 30) -> None:
    """Download a remote media URL directly to disk."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        with open(dest_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


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
            "ignore_no_formats_error": True,
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

            entries = list(info.get("entries") or [])
            is_gallery = bool(entries)
            gallery_count = len(entries) if is_gallery else 0

            return MediaMetadata(
                title=info.get("title") or "media",
                artist=clean_artist_tag(
                    artist=info.get("artist"),
                    artists=info.get("artists"),
                    uploader=info.get("uploader") or info.get("creator"),
                ),
                track=info.get("track") or info.get("title"),
                album=info.get("album"),
                duration=info.get("duration"),
                thumbnail=info.get("thumbnail"),
                width=info.get("width"),
                height=info.get("height"),
                filesize=info.get("filesize") or info.get("filesize_approx"),
                extractor=info.get("extractor") or "",
                formats=info.get("formats") or [],
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
                        duration=info.get("duration"),
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
                    duration=info.get("duration"),
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

            clean_artist = clean_artist_tag(
                artist=info.get("artist"),
                artists=info.get("artists"),
                uploader=info.get("uploader") or info.get("creator"),
            )
            clean_track = info.get("track") or info.get("title")

            # For audio downloads, rename output file to clean 'Artist - Track.ext'
            if file_path.suffix.lower() in {".mp3", ".m4a", ".flac", ".ogg", ".opus"}:
                clean_title_part = clean_track or info.get("title") or "Audio"
                if clean_artist:
                    new_filename = f"{clean_artist} - {clean_title_part}{file_path.suffix.lower()}"
                else:
                    new_filename = f"{clean_title_part}{file_path.suffix.lower()}"
                safe_name = re.sub(r'[\\/*?:"<>|]', "", new_filename)
                renamed_path = temp_dir / safe_name
                if renamed_path != file_path and not renamed_path.exists():
                    try:
                        file_path.rename(renamed_path)
                        file_path = renamed_path
                    except OSError:
                        pass

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
                            ["ffmpeg", "-y", "-i", str(webp_file), str(jpg_target)],
                            check=False,
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
