import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MediaMetadata:
    """Standardized metadata retrieved prior to downloading."""

    title: str
    duration: int | None = None
    thumbnail: str | None = None
    width: int | None = None
    height: int | None = None
    filesize: int | None = None
    extractor: str = ""
    formats: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DownloadedMedia:
    """Container for downloaded media files with deterministic cleanup."""

    filepath: Path
    title: str
    duration: int | None = None
    thumbnail_path: Path | None = None
    width: int | None = None
    height: int | None = None
    filesize: int = 0
    temp_dir: Path | None = None

    def cleanup(self) -> None:
        """Deterministically remove temporary media files and directories from disk."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug("Successfully purged temporary directory: %s", self.temp_dir)
            except Exception as e:
                logger.warning("Error purging temporary directory %s: %s", self.temp_dir, e)
        elif self.filepath and self.filepath.exists():
            try:
                self.filepath.unlink(missing_ok=True)
                logger.debug("Successfully purged temporary file: %s", self.filepath)
            except Exception as e:
                logger.warning("Error purging temporary file %s: %s", self.filepath, e)

    async def __aenter__(self) -> "DownloadedMedia":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
