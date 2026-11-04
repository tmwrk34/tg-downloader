"""
Data structures, enums, and models for Texas Hold'em game states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from services.games.poker.cards import Card


class PlayerStatus(StrEnum):
    ACTIVE = "active"  # In hand, waiting for or having made moves
    FOLDED = "folded"  # Folded during the current hand
    ALL_IN = "all_in"  # Committed all chips, waiting for showdown
    SITTING_OUT = "sitting_out"  # At the table but not participating in current hand


class GameStage(StrEnum):
    WAITING = "waiting"  # Waiting for players or start command
    PREFLOP = "preflop"  # 2 hole cards dealt, 1st betting round
    FLOP = "flop"  # 3 community cards dealt, 2nd betting round
    TURN = "turn"  # 4th community card dealt, 3rd betting round
    RIVER = "river"  # 5th community card dealt, final betting round
    SHOWDOWN = "showdown"  # Evaluation of hands and pot distribution
    HAND_ENDED = "ended"  # Brief cooldown before next hand


class MoveType(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class Player:
    user_id: int
    display_name: str
    username: str | None = None
    avatar_url: str | None = None
    chips: int = 1000
    seat_index: int = 0
    status: PlayerStatus = PlayerStatus.ACTIVE
    hole_cards: list[Card] = field(default_factory=list)
    current_bet: int = 0  # Chips bet in the current betting round
    total_hand_bet: int = 0  # Total chips invested into the whole hand (for pot split)

    def reset_for_hand(self) -> None:
        self.hole_cards = []
        self.current_bet = 0
        self.total_hand_bet = 0
        if self.chips > 0 and self.status != PlayerStatus.SITTING_OUT:
            self.status = PlayerStatus.ACTIVE
        elif self.chips == 0:
            self.status = PlayerStatus.SITTING_OUT

    def to_dict(self, include_private_cards: bool = False) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "username": self.username,
            "chips": self.chips,
            "seat_index": self.seat_index,
            "status": self.status.value,
            "current_bet": self.current_bet,
            "total_hand_bet": self.total_hand_bet,
            "has_cards": len(self.hole_cards) == 2,
            "hole_cards": [c.display for c in self.hole_cards] if include_private_cards else None,
        }


@dataclass
class TableConfig:
    max_seats: int = 6
    small_blind: int = 10
    big_blind: int = 20
    starting_chips: int = 1000
    turn_timeout_seconds: int = 45
    min_players_to_start: int = 2


@dataclass
class HandWinner:
    player_id: int
    display_name: str
    amount_won: int
    hand_description_uk: str
    best_cards: list[str] = field(default_factory=list)
