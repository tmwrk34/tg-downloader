from collections.abc import Iterable

from downloader.core import MediaDownloaderCore
from downloader.platforms.base import BasePlatformService
from downloader.platforms.instagram import InstagramService
from downloader.platforms.pinterest import PinterestService
from downloader.platforms.tiktok import TikTokService
from downloader.platforms.youtube import YouTubeService


class PlatformServiceRegistry:
    """Registry maintaining available platform service adapters."""

    def __init__(
        self,
        core: MediaDownloaderCore,
        services: Iterable[BasePlatformService] | None = None,
    ) -> None:
        self.core = core
        self._services: dict[str, BasePlatformService] = {}

        if services:
            for s in services:
                self.register(s)
        else:
            self.register_defaults()

    def register(self, service: BasePlatformService) -> None:
        """Register a new platform adapter."""
        self._services[service.platform_name.lower()] = service

    def get_service_by_name(self, name: str) -> BasePlatformService | None:
        """Retrieve an adapter by its platform name (case-insensitive)."""
        return self._services.get(name.lower())

    def get_service_for_url(self, url: str) -> BasePlatformService | None:
        """Find the matching platform adapter for a given URL."""
        for service in self._services.values():
            if service.is_supported(url):
                return service
        return None

    def register_defaults(self) -> None:
        """Register default platform adapters."""
        self.register(TikTokService(self.core))
        self.register(InstagramService(self.core))
        self.register(YouTubeService(self.core))
        self.register(PinterestService(self.core))

    def all_services(self) -> list[BasePlatformService]:
        """List all registered service adapters."""
        return list(self._services.values())
