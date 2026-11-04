"""
High-performance 7-card Texas Hold'em hand evaluator.
Supports all poker hand rankings with exact tie-breaking, card extractions,
and localized Ukrainian/English human-readable combination names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations

from services.games.poker.cards import Card


class HandRank(IntEnum):
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

    @property
    def label_uk(self) -> str:
        names = {
            HandRank.HIGH_CARD: "Старша карта",
            HandRank.ONE_PAIR: "Пара",
            HandRank.TWO_PAIR: "Дві пари",
            HandRank.THREE_OF_A_KIND: "Сет (Трійка)",
            HandRank.STRAIGHT: "Стріт",
            HandRank.FLUSH: "Флеш",
            HandRank.FULL_HOUSE: "Фул-хаус",
            HandRank.FOUR_OF_A_KIND: "Каре",
            HandRank.STRAIGHT_FLUSH: "Стріт-флеш",
            HandRank.ROYAL_FLUSH: "Роял-флеш",
        }
        return names[self]


@dataclass(frozen=True)
class HandEvaluation:
    rank: HandRank
    tie_breakers: tuple[int, ...]
    best_cards: tuple[Card, ...]
    description_uk: str

    def __lt__(self, other: HandEvaluation) -> bool:
        if not isinstance(other, HandEvaluation):
            return NotImplemented
        return (self.rank, self.tie_breakers) < (other.rank, other.tie_breakers)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandEvaluation):
            return False
        return (self.rank, self.tie_breakers) == (other.rank, other.tie_breakers)

    def __gt__(self, other: HandEvaluation) -> bool:
        if not isinstance(other, HandEvaluation):
            return NotImplemented
        return (self.rank, self.tie_breakers) > (other.rank, other.tie_breakers)


def _rank_name_uk(rank_val: int) -> str:
    names = {
        14: "Тузів",
        13: "Королів",
        12: "Дам",
        11: "Валетів",
        10: "Десяток",
        9: "Дев'яток",
        8: "Вісімок",
        7: "Сімок",
        6: "Шісток",
        5: "П'ятірок",
        4: "Четвірок",
        3: "Трійок",
        2: "Двійок",
    }
    return names.get(rank_val, str(rank_val))


def _rank_single_uk(rank_val: int) -> str:
    names = {
        14: "Туз",
        13: "Король",
        12: "Дама",
        11: "Валет",
        10: "Десятка",
        9: "Дев'ятка",
        8: "Вісімка",
        7: "Сімка",
        6: "Шістка",
        5: "П'ятірка",
        4: "Четвірка",
        3: "Трійка",
        2: "Двійка",
    }
    return names.get(rank_val, str(rank_val))


def _evaluate_5_cards(cards: Sequence[Card]) -> HandEvaluation:
    """Evaluate exactly 5 cards."""
    if len(cards) != 5:
        raise ValueError("Must provide exactly 5 cards to evaluate")

    sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
    ranks = [c.rank.value for c in sorted_cards]
    suits = [c.suit for c in sorted_cards]

    is_flush = len(set(suits)) == 1

    # Check for straight
    is_straight = False
    straight_high = 0

    # Normal straight check
    if len(set(ranks)) == 5 and (ranks[0] - ranks[4] == 4):
        is_straight = True
        straight_high = ranks[0]
    # Ace-low wheel check (A, 5, 4, 3, 2)
    elif ranks == [14, 5, 4, 3, 2]:
        is_straight = True
        straight_high = 5
        # Reorder Ace to end for best_cards display
        sorted_cards = sorted_cards[1:] + [sorted_cards[0]]

    # Straight flush / Royal flush
    if is_flush and is_straight:
        if straight_high == 14:
            return HandEvaluation(
                rank=HandRank.ROYAL_FLUSH,
                tie_breakers=(14,),
                best_cards=tuple(sorted_cards),
                description_uk="Роял-флеш",
            )
        return HandEvaluation(
            rank=HandRank.STRAIGHT_FLUSH,
            tie_breakers=(straight_high,),
            best_cards=tuple(sorted_cards),
            description_uk=f"Стріт-флеш до {_rank_single_uk(straight_high)}а",
        )

    # Rank count frequencies
    counts = Counter(ranks)
    by_freq = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    # by_freq looks like: [(rank, count), ...]

    # Four of a kind
    if by_freq[0][1] == 4:
        quad_rank = by_freq[0][0]
        kicker = by_freq[1][0]
        quad_cards = [c for c in sorted_cards if c.rank == quad_rank]
        kicker_cards = [c for c in sorted_cards if c.rank != quad_rank]
        return HandEvaluation(
            rank=HandRank.FOUR_OF_A_KIND,
            tie_breakers=(quad_rank, kicker),
            best_cards=tuple(quad_cards + kicker_cards),
            description_uk=f"Каре {_rank_name_uk(quad_rank)}",
        )

    # Full House
    if by_freq[0][1] == 3 and by_freq[1][1] == 2:
        trips_rank = by_freq[0][0]
        pair_rank = by_freq[1][0]
        trips_cards = [c for c in sorted_cards if c.rank == trips_rank]
        pair_cards = [c for c in sorted_cards if c.rank == pair_rank]
        return HandEvaluation(
            rank=HandRank.FULL_HOUSE,
            tie_breakers=(trips_rank, pair_rank),
            best_cards=tuple(trips_cards + pair_cards),
            description_uk=f"Фул-хаус ({_rank_name_uk(trips_rank)} і {_rank_name_uk(pair_rank)})",
        )

    # Flush
    if is_flush:
        return HandEvaluation(
            rank=HandRank.FLUSH,
            tie_breakers=tuple(ranks),
            best_cards=tuple(sorted_cards),
            description_uk=f"Флеш (старша {_rank_single_uk(ranks[0])})",
        )

    # Straight
    if is_straight:
        return HandEvaluation(
            rank=HandRank.STRAIGHT,
            tie_breakers=(straight_high,),
            best_cards=tuple(sorted_cards),
            description_uk=f"Стріт до {_rank_single_uk(straight_high)}а",
        )

    # Three of a Kind
    if by_freq[0][1] == 3:
        trips_rank = by_freq[0][0]
        kickers = [item[0] for item in by_freq[1:]]
        trips_cards = [c for c in sorted_cards if c.rank == trips_rank]
        kicker_cards = [c for c in sorted_cards if c.rank != trips_rank]
        return HandEvaluation(
            rank=HandRank.THREE_OF_A_KIND,
            tie_breakers=(trips_rank, *kickers),
            best_cards=tuple(trips_cards + kicker_cards),
            description_uk=f"Сет {_rank_name_uk(trips_rank)}",
        )

    # Two Pair
    if by_freq[0][1] == 2 and by_freq[1][1] == 2:
        high_pair = max(by_freq[0][0], by_freq[1][0])
        low_pair = min(by_freq[0][0], by_freq[1][0])
        kicker = by_freq[2][0]
        pair_high_cards = [c for c in sorted_cards if c.rank == high_pair]
        pair_low_cards = [c for c in sorted_cards if c.rank == low_pair]
        kicker_cards = [c for c in sorted_cards if c.rank not in (high_pair, low_pair)]
        return HandEvaluation(
            rank=HandRank.TWO_PAIR,
            tie_breakers=(high_pair, low_pair, kicker),
            best_cards=tuple(pair_high_cards + pair_low_cards + kicker_cards),
            description_uk=f"Дві пари ({_rank_name_uk(high_pair)} і {_rank_name_uk(low_pair)})",
        )

    # One Pair
    if by_freq[0][1] == 2:
        pair_rank = by_freq[0][0]
        kickers = [item[0] for item in by_freq[1:]]
        pair_cards = [c for c in sorted_cards if c.rank == pair_rank]
        kicker_cards = [c for c in sorted_cards if c.rank != pair_rank]
        return HandEvaluation(
            rank=HandRank.ONE_PAIR,
            tie_breakers=(pair_rank, *kickers),
            best_cards=tuple(pair_cards + kicker_cards),
            description_uk=f"Пара {_rank_name_uk(pair_rank)}",
        )

    # High Card
    return HandEvaluation(
        rank=HandRank.HIGH_CARD,
        tie_breakers=tuple(ranks),
        best_cards=tuple(sorted_cards),
        description_uk=f"Старша карта {_rank_single_uk(ranks[0])}",
    )


def evaluate_hand(cards: Sequence[Card]) -> HandEvaluation:
    """
    Evaluate poker hand from 5, 6, or 7 cards (best 5-card subset).
    """
    n = len(cards)
    if n < 5:
        raise ValueError(f"Need at least 5 cards to evaluate, got {n}")
    if n == 5:
        return _evaluate_5_cards(cards)

    # 6 or 7 cards: find the maximum 5-card combination
    best_eval: HandEvaluation | None = None
    for combo in combinations(cards, 5):
        current_eval = _evaluate_5_cards(combo)
        if best_eval is None or current_eval > best_eval:
            best_eval = current_eval

    assert best_eval is not None
    return best_eval
