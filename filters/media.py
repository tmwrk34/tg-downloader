import re
from typing import Any

from aiogram.filters import BaseFilter
from aiogram.types import Message

from filters.patterns import MEDIA_PATTERNS, Platform, clean_url


class MediaUrlFilter(BaseFilter):
    """Filter that detects and extracts media URLs from message text or caption.

    Extracts supported URLs (TikTok, Instagram, YouTube) and injects:
    - `url`: cleaned extracted URL string
    - `platform`: platform name ('tiktok', 'instagram', 'youtube')
    - `media_url`: alias to `url` for convenience
    """

    def __init__(
        self,
        *platforms: Platform | str,
        patterns: dict[Platform, re.Pattern[str]] | None = None,
    ) -> None:
        self.patterns = patterns or MEDIA_PATTERNS
        if platforms:
            target_platforms: set[Platform] = set()
            for p in platforms:
                if isinstance(p, Platform):
                    target_platforms.add(p)
                else:
                    target_platforms.add(Platform(p.lower()))
            self.target_platforms = target_platforms
        else:
            self.target_platforms = set(self.patterns.keys())

    async def __call__(self, message: Message) -> bool | dict[str, Any]:
        text = message.text or message.caption
        if not text:
            return False

        for platform in self.target_platforms:
            pattern = self.patterns.get(platform)
            if not pattern:
                continue

            match = pattern.search(text)
            if match:
                raw_url = match.group(0)
                extracted_url = clean_url(raw_url)
                return {
                    "url": extracted_url,
                    "platform": platform.value,
                    "media_url": extracted_url,
                }

        return False
