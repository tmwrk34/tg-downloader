import re
from collections.abc import Iterable
from enum import StrEnum


class Platform(StrEnum):
    """Supported media platforms."""

    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    PINTEREST = "pinterest"


# Compiled regular expressions for supported media platforms.
# Patterns match the URL structure within arbitrary text.
MEDIA_PATTERNS: dict[Platform, re.Pattern[str]] = {
    Platform.TIKTOK: re.compile(
        r"https?://(?:(?:www|m|v)\.)?tiktok\.com/(?:@[a-zA-Z0-9_.-]+/(?:video|photo)/\d+|t/[a-zA-Z0-9_-]+|v/\d+(?:\.html)?)[^\s<>\"']*"
        r"|https?://(?:vm|vt)\.tiktok\.com/[a-zA-Z0-9_-]+[^\s<>\"']*",
        re.IGNORECASE,
    ),
    Platform.INSTAGRAM: re.compile(
        r"https?://(?:www\.)?(?:instagram\.com|instagr\.am)/(?:p|reel|reels|tv|stories/[a-zA-Z0-9_.-]+|share/(?:reel|p))/[a-zA-Z0-9_-]+[^\s<>\"']*",
        re.IGNORECASE,
    ),
    Platform.YOUTUBE: re.compile(
        r"https?://(?:(?:www|m|music)\.)?youtube\.com/(?:watch\?[^\s<>\"']*v=[a-zA-Z0-9_-]+|shorts/[a-zA-Z0-9_-]+|live/[a-zA-Z0-9_-]+|embed/[a-zA-Z0-9_-]+)[^\s<>\"']*"
        r"|https?://youtu\.be/[a-zA-Z0-9_-]+[^\s<>\"']*",
        re.IGNORECASE,
    ),
    Platform.PINTEREST: re.compile(
        r"https?://(?:(?:www|[a-z]{2})\.)?pinterest\.(?:com|co\.[a-z]{2}|[a-z]{2})/pin/(?:[\w-]+--)?\d+[^\s<>\"']*"
        r"|https?://(?:www\.)?pin\.it/[a-zA-Z0-9]+[^\s<>\"']*",
        re.IGNORECASE,
    ),
}

# Trailing characters to strip from extracted URLs (punctuation attached at sentence end).
TRAILING_PUNCTUATION = ".,!?:;)'\"]}>"
_PLAYLIST_PARAM_RE = re.compile(r"&(?:list|index)=[^&\s]+")


def clean_url(raw_url: str) -> str:
    """Strip extraneous sentence-ending punctuation and playlist query tracking from extracted URLs."""
    url = raw_url.strip()
    while url and url[-1] in TRAILING_PUNCTUATION:
        # If closing parenthesis or bracket, only strip if unmatched
        if url[-1] == ")" and url.count("(") >= url.count(")"):
            break
        if url[-1] == "]" and url.count("[") >= url.count("]"):
            break
        url = url[:-1]

    # For single video/music links, strip playlist parameters to prevent slow playlist downloads
    if "/watch?" in url:
        url = _PLAYLIST_PARAM_RE.sub("", url)

    return url


class TriggerRegistry:
    """Centralized registry for constant expression trigger phrases."""

    def __init__(self, initial_triggers: dict[str, set[str]] | None = None) -> None:
        self._registry: dict[str, set[str]] = {}
        if initial_triggers:
            for category, phrases in initial_triggers.items():
                self.register(category, phrases)

    def register(self, category: str, phrases: Iterable[str]) -> None:
        """Register one or more trigger phrases under a category."""
        key = category.strip().lower()
        if key not in self._registry:
            self._registry[key] = set()
        for phrase in phrases:
            cleaned = phrase.strip()
            if cleaned:
                self._registry[key].add(cleaned)

    def get(self, category: str) -> set[str]:
        """Get a copy of trigger phrases for a category."""
        return self._registry.get(category.strip().lower(), set()).copy()

    def all_categories(self) -> list[str]:
        """List all registered trigger categories."""
        return list(self._registry.keys())

    def all_phrases(self) -> set[str]:
        """Get a combined set of all registered phrases across categories."""
        combined: set[str] = set()
        for phrases in self._registry.values():
            combined.update(phrases)
        return combined


# Default pre-configured trigger registry instance
default_trigger_registry = TriggerRegistry(
    initial_triggers={
        "ping": {"ping", "pong"},
        "help": {"help", "info"},
    }
)
