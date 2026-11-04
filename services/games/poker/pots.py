"""
Main pot and side pot calculations for Texas Hold'em.
Accurately splits bets into nested pots based on players' committed amounts
and all-in thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pot:
    amount: int
    eligible_player_ids: set[int] = field(default_factory=set)


def calculate_pots(player_bets: dict[int, int], active_player_ids: set[int]) -> list[Pot]:
    """
    Given a map of {player_id: total_committed_chips} and set of players who haven't folded,
    compute all main and side pots in order.

    - player_bets includes all contributions across the entire hand (even from folded players).
    - active_player_ids includes only players who did NOT fold and are eligible to win pots.
    """
    # Only consider players who committed > 0
    contributions = {pid: amt for pid, amt in player_bets.items() if amt > 0}
    if not contributions:
        return []

    pots: list[Pot] = []
    # Distinct positive contribution levels sorted ascending
    levels = sorted({amt for amt in contributions.values() if amt > 0})

    prev_level = 0
    for level in levels:
        tier_amount = level - prev_level
        pot_chips = 0
        eligible: set[int] = set()

        for pid, amt in contributions.items():
            if amt >= level:
                pot_chips += tier_amount
                if pid in active_player_ids:
                    eligible.add(pid)
            elif amt > prev_level:
                partial = amt - prev_level
                pot_chips += partial
                if pid in active_player_ids:
                    eligible.add(pid)

        if pot_chips > 0:
            if pots and pots[-1].eligible_player_ids == eligible:
                # Merge with previous pot if exact same eligible player set
                pots[-1].amount += pot_chips
            else:
                pots.append(Pot(amount=pot_chips, eligible_player_ids=eligible))

        prev_level = level

    return pots
