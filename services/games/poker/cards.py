"""
Poker card, suit, rank, and deck definitions with cryptographically secure shuffling.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import IntEnum, StrEnum


class Suit(StrEnum):
    SPADES = "♠"
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def symbol(self) -> str:
        symbols = {
            Rank.TWO: "2",
            Rank.THREE: "3",
            Rank.FOUR: "4",
            Rank.FIVE: "5",
            Rank.SIX: "6",
            Rank.SEVEN: "7",
            Rank.EIGHT: "8",
            Rank.NINE: "9",
            Rank.TEN: "10",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }
        return symbols[self]


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def code(self) -> str:
        """Compact string representation, e.g. 'As', 'Kh', '10d', '2c'."""
        suit_code = {
            Suit.SPADES: "s",
            Suit.HEARTS: "h",
            Suit.DIAMONDS: "d",
            Suit.CLUBS: "c",
        }[self.suit]
        return f"{self.rank.symbol}{suit_code}"

    @property
    def display(self) -> str:
        """User-friendly display string, e.g. 'A♠', '10♦'."""
        return f"{self.rank.symbol}{self.suit.value}"

    @property
    def is_red(self) -> bool:
        return self.suit in (Suit.HEARTS, Suit.DIAMONDS)

    @classmethod
    def from_str(cls, text: str) -> Card:
        """Parse card from notation like 'As', '10d', 'A♠', 'K♥'."""
        t = text.strip()
        if not t:
            raise ValueError("Empty card string")
        suit_map = {
            "s": Suit.SPADES,
            "♠": Suit.SPADES,
            "h": Suit.HEARTS,
            "♥": Suit.HEARTS,
            "d": Suit.DIAMONDS,
            "♦": Suit.DIAMONDS,
            "c": Suit.CLUBS,
            "♣": Suit.CLUBS,
        }
        last_char = t[-1].lower()
        if last_char not in suit_map:
            raise ValueError(f"Invalid suit in '{text}'")
        suit = suit_map[last_char]

        rank_str = t[:-1].upper()
        rank_map = {
            "2": Rank.TWO,
            "3": Rank.THREE,
            "4": Rank.FOUR,
            "5": Rank.FIVE,
            "6": Rank.SIX,
            "7": Rank.SEVEN,
            "8": Rank.EIGHT,
            "9": Rank.NINE,
            "10": Rank.TEN,
            "T": Rank.TEN,
            "J": Rank.JACK,
            "Q": Rank.QUEEN,
            "K": Rank.KING,
            "A": Rank.ACE,
        }
        if rank_str not in rank_map:
            raise ValueError(f"Invalid rank in '{text}'")
        return cls(rank=rank_map[rank_str], suit=suit)


class Deck:
    """Standard 52-card deck using cryptographically secure randomization."""

    def __init__(self) -> None:
        self._cards: list[Card] = []
        self.reset()

    def reset(self) -> None:
        self._cards = [Card(rank=r, suit=s) for r in Rank for s in Suit]
        self.shuffle()

    def shuffle(self) -> None:
        """Cryptographically secure Fisher-Yates shuffle."""
        rng = secrets.SystemRandom()
        rng.shuffle(self._cards)

    def draw(self, count: int = 1) -> list[Card]:
        if count < 1:
            return []
        if len(self._cards) < count:
            raise ValueError(
                f"Not enough cards in deck to draw {count} (remaining: {len(self._cards)})"
            )
        drawn = self._cards[-count:]
        del self._cards[-count:]
        return drawn

    def draw_one(self) -> Card:
        cards = self.draw(1)
        return cards[0]

    @property
    def remaining(self) -> int:
        return len(self._cards)
