from collections.abc import Iterable
from typing import Any

from aiogram.filters import BaseFilter
from aiogram.types import Message

from filters.patterns import TriggerRegistry, default_trigger_registry


class TriggerPhraseFilter(BaseFilter):
    """Filter that matches message text or caption against predefined trigger phrases.

    Supports:
    - Dedicated configuration via `TriggerRegistry` or ad-hoc phrases.
    - Exact match vs. substring match.
    - Case-insensitive matching by default.
    - Injects `matched_phrase` and `trigger` into handler arguments.
    """

    def __init__(
        self,
        phrases: Iterable[str] | None = None,
        category: str | None = None,
        registry: TriggerRegistry | None = None,
        exact_match: bool = False,
        case_sensitive: bool = False,
    ) -> None:
        self.exact_match = exact_match
        self.case_sensitive = case_sensitive
        self._registry = registry or default_trigger_registry

        if phrases is not None:
            self._phrases = {p.strip() for p in phrases if p.strip()}
        elif category is not None:
            self._phrases = self._registry.get(category)
        else:
            self._phrases = self._registry.all_phrases()

    async def __call__(self, message: Message) -> bool | dict[str, Any]:
        text = message.text or message.caption
        if not text:
            return False

        normalized_text = text if self.case_sensitive else text.casefold()
        trimmed_text = normalized_text.strip()

        for raw_phrase in self._phrases:
            phrase_to_match = raw_phrase if self.case_sensitive else raw_phrase.casefold()

            if self.exact_match:
                if trimmed_text == phrase_to_match:
                    return {
                        "matched_phrase": raw_phrase,
                        "trigger": raw_phrase,
                    }
            else:
                if phrase_to_match in normalized_text:
                    return {
                        "matched_phrase": raw_phrase,
                        "trigger": raw_phrase,
                    }

        return False
