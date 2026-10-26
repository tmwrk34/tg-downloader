from __future__ import annotations

import sys
from typing import Any


class _BotProxy:
    """
    Lazy proxy to the root 'bot' module to completely eliminate circular imports
    during startup, while preserving dynamic attribute access (e.g. for mocks in tests).
    """

    def __getattr__(self, name: str) -> Any:
        mod = sys.modules.get("bot")
        if mod is None:
            import bot

            mod = bot
        return getattr(mod, name)


bot_proxy = _BotProxy()
