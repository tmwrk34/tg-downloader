from typing import Any

from aiogram.filters import BaseFilter
from aiogram.types import Message

from services.dice import DiceParser


class DiceFilter(BaseFilter):
    """Filter that matches natural language or plain-text dice expressions."""

    async def __call__(self, message: Message) -> bool | dict[str, Any]:
        text = message.text or message.caption
        if not text:
            return False

        stripped = text.strip()
        # Skip standard slash-commands (they are handled by Command filters)
        if stripped.startswith("/"):
            return False

        if DiceParser.is_dice_expression(stripped):
            norm_expr, is_secret = DiceParser.normalize_input(stripped)
            return {"dice_expr": norm_expr, "is_secret": is_secret}

        return False
