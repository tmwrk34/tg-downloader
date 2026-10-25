import random
import re
import time
import uuid
from dataclasses import dataclass, field

# Cryptographically secure pseudo-random number generator
_rng = random.SystemRandom()

MAX_DICE_COUNT = 100
MAX_DICE_SIDES = 1000
MAX_EXPRESSION_TERMS = 20


@dataclass
class SingleDiceRoll:
    """Individual die roll result with kept/dropped status."""

    value: int
    is_kept: bool = True


@dataclass
class DiceTermResult:
    """Evaluation result of a single dice term (e.g. 4d6k3)."""

    count: int
    sides: int
    rolls: list[SingleDiceRoll]
    sign: int = 1
    modifier_type: str | None = None
    modifier_count: int | None = None

    @property
    def total(self) -> int:
        return sum(r.value for r in self.rolls if r.is_kept) * self.sign

    @property
    def kept_values(self) -> list[int]:
        return [r.value for r in self.rolls if r.is_kept]

    def format_details(self) -> str:
        parts = []
        for r in self.rolls:
            if r.is_kept:
                parts.append(str(r.value))
            else:
                parts.append(f"<s>{r.value}</s>")
        return f"[{', '.join(parts)}]"


@dataclass
class DiceRollResult:
    """Complete evaluation result for a dice roll expression."""

    expression: str
    normalized_expression: str
    dice_terms: list[DiceTermResult] = field(default_factory=list)
    constant_modifier: int = 0
    total: int = 0
    is_crit_hit: bool = False
    is_crit_fail: bool = False

    def format_html(self, user_name: str | None = None) -> str:
        """Format roll result as a clean, structured HTML message in English."""
        header = f"🎲 <b>{user_name}'s Roll:</b>" if user_name else "🎲 <b>Dice Roll:</b>"
        expr_badge = f"<code>{self.normalized_expression}</code>"

        details_parts = []
        for idx, term in enumerate(self.dice_terms):
            prefix = ""
            if idx > 0:
                prefix = " + " if term.sign > 0 else " - "
            details_parts.append(f"{prefix}{term.format_details()}")

        if self.constant_modifier != 0:
            mod_sign = " + " if self.constant_modifier > 0 else " - "
            details_parts.append(f"{mod_sign}{abs(self.constant_modifier)}")

        details_str = "".join(details_parts) if details_parts else str(self.total)

        lines = [
            f"{header} {expr_badge}",
            f"🎯 <b>Result:</b> <b>{self.total}</b>",
            f"📝 <b>Details:</b> {details_str}",
        ]

        if self.is_crit_hit:
            lines.append("✨ <b>CRITICAL HIT! (Natural 20)</b> 🎯")
        elif self.is_crit_fail:
            lines.append("💀 <b>CRITICAL FAIL! (Natural 1)</b> 💥")

        return "\n".join(lines)

    def format_plain_text(self, user_name: str | None = None) -> str:
        """Format roll result as clean plain text without HTML tags for Telegram alerts."""
        header = f"🎲 {user_name}'s Secret Roll:" if user_name else "🎲 Secret Roll:"
        expr_badge = self.normalized_expression

        details_parts = []
        for idx, term in enumerate(self.dice_terms):
            prefix = ""
            if idx > 0:
                prefix = " + " if term.sign > 0 else " - "
            term_str = ", ".join(str(r.value) if r.is_kept else f"({r.value})" for r in term.rolls)
            details_parts.append(f"{prefix}[{term_str}]")

        if self.constant_modifier != 0:
            mod_sign = " + " if self.constant_modifier > 0 else " - "
            details_parts.append(f"{mod_sign}{abs(self.constant_modifier)}")

        details_str = "".join(details_parts) if details_parts else str(self.total)

        lines = [
            f"{header} {expr_badge}",
            f"🎯 Result: {self.total}",
            f"📝 Details: {details_str}",
        ]

        if self.is_crit_hit:
            lines.append("✨ CRITICAL HIT! (Natural 20) 🎯")
        elif self.is_crit_fail:
            lines.append("💀 CRITICAL FAIL! (Natural 1) 💥")

        return "\n".join(lines)


# In-memory secret rolls store: secret_id -> (roller_user_id, roller_name, result_html, result_plain, created_at)
MAX_SECRET_ROLLS_CACHE_SIZE = 500
_secret_rolls_cache: dict[str, tuple[int, str, str, str, float]] = {}


def store_secret_roll(user_id: int, user_name: str, result_html: str, result_plain: str) -> str:
    """Store secret roll in memory and return its lookup UUID token."""
    secret_id = uuid.uuid4().hex[:12]
    now = time.monotonic()
    # Prune expired entries older than 24h (86400s)
    expired_keys = [k for k, (_, _, _, _, ts) in _secret_rolls_cache.items() if now - ts > 86400]
    for k in expired_keys:
        _secret_rolls_cache.pop(k, None)

    # Enforce strict upper bound on cache size
    while len(_secret_rolls_cache) >= MAX_SECRET_ROLLS_CACHE_SIZE:
        oldest_key = next(iter(_secret_rolls_cache))
        _secret_rolls_cache.pop(oldest_key, None)

    _secret_rolls_cache[secret_id] = (user_id, user_name, result_html, result_plain, now)
    return secret_id


def get_secret_roll(secret_id: str) -> tuple[int, str, str, str] | None:
    """Retrieve secret roll metadata: (user_id, user_name, result_html, result_plain) if exists."""
    entry = _secret_rolls_cache.get(secret_id)
    if not entry:
        return None
    user_id, user_name, result_html, result_plain, _ = entry
    return user_id, user_name, result_html, result_plain


class DiceParser:
    """Parser and evaluator for tabletop dice expressions."""

    # Matches dice expressions like 2d6, 2д20, 2к20, 4d6k3, 4к6к3, 2d20kh1, d20, д20, к20
    DICE_PATTERN = re.compile(
        r"^(\d+)?([dдк])(\d+)(?:([a-zа-яёіїє]+)(\d+))?$",
        re.IGNORECASE,
    )
    SECRET_PREFIX_PATTERNS = (
        re.compile(r"^(?:/dmroll|dmroll|/дмролл|дмролл|/дм|дм)\b", re.IGNORECASE),
        re.compile(
            r"^(?:dm\s*roll|hidden\s*roll|secret\s*roll|дм\s*ролл|секретний\s*ролл|секретный\s*ролл)\b",
            re.IGNORECASE,
        ),
    )
    ROLL_CMD_PATTERN = re.compile(r"^(?:/roll|/ролл|/кинь|/кидок)\b", re.IGNORECASE)
    ACTION_VERB_PATTERN = re.compile(
        r"^(?:roll|throw|кинь|ролл|кидок|брось|бросить)\s+", re.IGNORECASE
    )
    SIDED_DICE_PATTERN = re.compile(
        r"(\d+)\s*(?:sided\s*dice|гранний\s*кубик|гранный\s*кубик|граней)", re.IGNORECASE
    )
    PLUS_MOD_PATTERN = re.compile(r"([0-9a-zа-яёіїє])(?:p|п)([0-9]+)", re.IGNORECASE)
    MINUS_MOD_PATTERN = re.compile(r"([0-9a-zа-яёіїє])(?:m|м)([0-9]+)", re.IGNORECASE)
    WHITESPACE_OP_PATTERN = re.compile(r"\s*([+-])\s*")
    HAS_DICE_TERM_PATTERN = re.compile(r"(?:^|[^\w])\d*[dдк]\d+", re.IGNORECASE)
    TOKEN_SPLIT_PATTERN = re.compile(r"([+-]?)([^+-]+)")

    MODIFIER_MAP: dict[str, str] = {
        # Keep highest
        "k": "kh",
        "kh": "kh",
        "к": "kh",
        "кв": "kh",
        # Keep lowest
        "kl": "kl",
        "кн": "kl",
        # Drop lowest
        "d": "dl",
        "dl": "dl",
        "с": "dl",
        "сн": "dl",
        "д": "dl",
        "дн": "dl",
        # Drop highest
        "dh": "dh",
        "св": "dh",
        "дв": "dh",
    }

    @classmethod
    def normalize_input(cls, raw: str) -> tuple[str, bool]:
        """
        Normalize natural language and transliterated text into a clean dice expression.
        Returns: (normalized_expression, is_secret)
        """
        text = raw.strip()

        # Check for secret roll prefixes (English & Cyrillic)
        is_secret = False
        for pattern in cls.SECRET_PREFIX_PATTERNS:
            m = pattern.match(text)
            if m:
                is_secret = True
                text = text[m.end() :].strip()
                break

        # Standard command prefix /roll and Cyrillic aliases
        text = cls.ROLL_CMD_PATTERN.sub("", text).strip()

        # Natural language action verbs (English & Ukrainian/Russian)
        text = cls.ACTION_VERB_PATTERN.sub("", text).strip()

        # Natural language descriptive dice: "20 sided dice", "20 гранний кубик"
        text = cls.SIDED_DICE_PATTERN.sub(r"1d\1", text)

        # Text-based modifiers:
        # 2d6p3 / 2д6п3 -> 2d6+3
        # 2d6m3 / 2д6м3 -> 2d6-3
        text = cls.PLUS_MOD_PATTERN.sub(r"\1+\2", text)
        text = cls.MINUS_MOD_PATTERN.sub(r"\1-\2", text)

        # Remove extraneous whitespace around + and -
        text = cls.WHITESPACE_OP_PATTERN.sub(r"\1", text)

        return text, is_secret

    @classmethod
    def is_dice_expression(cls, raw: str) -> bool:
        """Check whether raw text can be interpreted as a dice roll."""
        cleaned, _ = cls.normalize_input(raw)
        if not cleaned or not cls.HAS_DICE_TERM_PATTERN.search(cleaned):
            return False

        try:
            cls.parse_and_roll(cleaned)
            return True
        except ValueError:
            return False

    @classmethod
    def parse_and_roll(cls, expr: str) -> DiceRollResult:
        """Parse dice expression and compute randomized roll results."""
        clean_expr, _ = cls.normalize_input(expr)
        if not clean_expr:
            raise ValueError("Dice expression is empty.")

        # Split into signed tokens (+term or -term)
        # E.g. "4d6k3+2" -> ["4d6k3", "+2"], "2d6-1d4-3" -> ["2d6", "-1d4", "-3"]
        token_matches = cls.TOKEN_SPLIT_PATTERN.finditer(clean_expr)

        dice_terms: list[DiceTermResult] = []
        constant_modifier = 0

        term_count = 0
        for match in token_matches:
            term_count += 1
            if term_count > MAX_EXPRESSION_TERMS:
                raise ValueError(f"Too many terms in expression (maximum {MAX_EXPRESSION_TERMS}).")

            sign_str, token = match.group(1), match.group(2).strip()
            sign = -1 if sign_str == "-" else 1

            if not token:
                continue

            # Check if token is a constant integer modifier
            if token.isdigit():
                constant_modifier += sign * int(token)
                continue

            # Otherwise, must be a dice term: e.g. 4d6k3, 2д20, 2к20, 4к6к3
            dice_match = cls.DICE_PATTERN.match(token)
            if not dice_match:
                raise ValueError(f"Invalid dice term: '{token}'")

            raw_count, _sep, raw_sides, raw_mod_type, raw_mod_count = dice_match.groups()

            count = int(raw_count) if raw_count else 1
            sides = int(raw_sides)

            if count <= 0 or count > MAX_DICE_COUNT:
                raise ValueError(f"Dice count must be between 1 and {MAX_DICE_COUNT}.")
            if sides <= 0 or sides > MAX_DICE_SIDES:
                raise ValueError(f"Dice sides must be between 1 and {MAX_DICE_SIDES}.")

            norm_mod_type = None
            mod_count = None
            if raw_mod_type:
                norm_mod_type = cls.MODIFIER_MAP.get(raw_mod_type.lower())
                if not norm_mod_type:
                    raise ValueError(f"Invalid keep/drop modifier: '{raw_mod_type}'")
                mod_count = int(raw_mod_count) if raw_mod_count else 1
                if mod_count <= 0 or mod_count > count:
                    raise ValueError(
                        f"Keep/drop count ({mod_count}) must be between 1 and total dice ({count})."
                    )

            # Roll the dice
            raw_rolls = [_rng.randint(1, sides) for _ in range(count)]
            single_rolls = [SingleDiceRoll(value=v, is_kept=True) for v in raw_rolls]

            # Apply keep / drop logic
            if norm_mod_type and mod_count is not None:
                indexed_rolls = list(enumerate(raw_rolls))

                if norm_mod_type == "kh":
                    # Keep N highest
                    indexed_rolls.sort(key=lambda x: x[1], reverse=True)
                    kept_indices = {idx for idx, _ in indexed_rolls[:mod_count]}
                    for i, r in enumerate(single_rolls):
                        r.is_kept = i in kept_indices
                elif norm_mod_type == "kl":
                    # Keep N lowest
                    indexed_rolls.sort(key=lambda x: x[1])
                    kept_indices = {idx for idx, _ in indexed_rolls[:mod_count]}
                    for i, r in enumerate(single_rolls):
                        r.is_kept = i in kept_indices
                elif norm_mod_type == "dl":
                    # Drop N lowest
                    indexed_rolls.sort(key=lambda x: x[1])
                    dropped_indices = {idx for idx, _ in indexed_rolls[:mod_count]}
                    for i, r in enumerate(single_rolls):
                        r.is_kept = i not in dropped_indices
                elif norm_mod_type == "dh":
                    # Drop N highest
                    indexed_rolls.sort(key=lambda x: x[1], reverse=True)
                    dropped_indices = {idx for idx, _ in indexed_rolls[:mod_count]}
                    for i, r in enumerate(single_rolls):
                        r.is_kept = i not in dropped_indices

            term_result = DiceTermResult(
                count=count,
                sides=sides,
                rolls=single_rolls,
                sign=sign,
                modifier_type=norm_mod_type,
                modifier_count=mod_count,
            )
            dice_terms.append(term_result)

        if not dice_terms and constant_modifier == 0:
            raise ValueError("No valid dice or modifiers provided.")

        total = sum(t.total for t in dice_terms) + constant_modifier

        # Canonical standardized expression (e.g. 2d20, 4d6kh3+2)
        canonical_parts = []
        for idx, t in enumerate(dice_terms):
            prefix = ""
            if idx > 0:
                prefix = "+" if t.sign > 0 else "-"
            elif t.sign < 0:
                prefix = "-"
            mod_str = (
                f"{t.modifier_type}{t.modifier_count}"
                if t.modifier_type and t.modifier_count
                else ""
            )
            canonical_parts.append(f"{prefix}{t.count}d{t.sides}{mod_str}")

        if constant_modifier != 0:
            mod_prefix = "+" if constant_modifier > 0 else "-"
            canonical_parts.append(f"{mod_prefix}{abs(constant_modifier)}")

        canonical_expr = "".join(canonical_parts) if canonical_parts else clean_expr

        # Check for Natural 20 / Natural 1 on d20 rolls
        is_crit_hit = False
        is_crit_fail = False

        # Only evaluate crit hit / fail if d20 dice were rolled
        d20_terms = [t for t in dice_terms if t.sides == 20]
        if d20_terms:
            kept_d20_values = [v for t in d20_terms for v in t.kept_values]
            if 20 in kept_d20_values:
                is_crit_hit = True
            elif all(v == 1 for v in kept_d20_values):
                is_crit_fail = True

        return DiceRollResult(
            expression=expr,
            normalized_expression=canonical_expr,
            dice_terms=dice_terms,
            constant_modifier=constant_modifier,
            total=total,
            is_crit_hit=is_crit_hit,
            is_crit_fail=is_crit_fail,
        )
