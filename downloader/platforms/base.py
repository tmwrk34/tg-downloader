from abc import ABC, abstractmethod
from typing import Any

from downloader.core import MediaDownloaderCore
from downloader.models import DownloadedMedia, MediaMetadata


class BasePlatformService(ABC):
    """Abstract base class for platform-specific media download adapters."""

    def __init__(self, core: MediaDownloaderCore) -> None:
        self.core = core

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Normalized platform name (e.g. 'tiktok', 'instagram', 'youtube')."""
        ...

    @abstractmethod
    def is_supported(self, url: str) -> bool:
        """Check whether the given URL is supported by this adapter."""
        ...

    @abstractmethod
    def get_ytdl_options(self) -> dict[str, Any]:
        """Return platform-specific yt-dlp configuration options."""
        ...

    async def extract_metadata(
        self,
        url: str,
        extra_opts: dict[str, Any] | None = None,
    ) -> MediaMetadata:
        """Extract metadata applying platform-specific options."""
        opts = self.get_ytdl_options()
        if extra_opts:
            opts.update(extra_opts)
        return await self.core.extract_metadata(url, custom_opts=opts)

    async def download(
        self,
        url: str,
        extra_opts: dict[str, Any] | None = None,
    ) -> DownloadedMedia:
        """Download media applying platform-specific options."""
        opts = self.get_ytdl_options()
        if extra_opts:
            opts.update(extra_opts)
        return await self.core.download(url, custom_opts=opts)
