"""
Tavern Games Table Registry.
Maintains active tables per chat_id in memory.
"""

from __future__ import annotations

import logging

from services.games.poker.engine import PokerTable
from services.games.poker.models import TableConfig

logger = logging.getLogger(__name__)


class GamesTableManager:
    """Manages active poker tables across chats."""

    def __init__(self) -> None:
        self._tables: dict[int, PokerTable] = {}

    def get_or_create_poker_table(
        self, chat_id: int, config: TableConfig | None = None
    ) -> PokerTable:
        if chat_id not in self._tables:
            self._tables[chat_id] = PokerTable(chat_id=chat_id, config=config)
            logger.info("Created new PokerTable for chat_id=%s", chat_id)
        return self._tables[chat_id]

    def get_poker_table(self, chat_id: int) -> PokerTable | None:
        return self._tables.get(chat_id)

    def close_table(self, chat_id: int) -> bool:
        table = self._tables.pop(chat_id, None)
        if table:
            table._cancel_turn_timer()
            return True
        return False


games_manager = GamesTableManager()
