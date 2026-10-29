"""
Persistent System Prompt Management for Munito AI Assistant.

Provides modular loading, updating, persistence, and factory reset
for the bot's system prompt across deployments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Base factory prompt template located in prompts/system_prompt.txt
DEFAULT_BASE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"

# Hardcoded emergency fallback prompt if template files are missing
FALLBACK_SYSTEM_PROMPT = (
    "You are Bard (@bard_taverna_bot), the tavern keeper and AI companion of this Telegram chat. "
    "Respond in the exact same language as the user's message with wit, conciseness, and warmth."
)


def resolve_custom_prompt_path(
    env_var: str = "SYSTEM_PROMPT_PATH",
    default_filename: str = "system_prompt.txt",
    test_marker: str = ".write_test_prompt",
) -> Path:
    """
    Resolve the persistent file path for custom system prompt overrides.

    Checks:
    1. Environment variable (e.g. `SYSTEM_PROMPT_PATH`)
    2. `data/<default_filename>` if writable or already existing (mounted Docker volume `./data`)
    3. `./<default_filename>` fallback
    """
    env_path = os.getenv(env_var)
    if env_path:
        return Path(env_path)

    data_dir = Path("data")
    data_file = data_dir / default_filename
    if data_file.exists():
        return data_file

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / test_marker
        test_file.touch()
        test_file.unlink()
        return data_file
    except (PermissionError, OSError):
        return Path(default_filename)


def load_system_prompt(
    custom_path: Path | str | None = None,
    base_path: Path | str | None = None,
) -> str:
    """
    Load active system prompt.

    Priority:
    1. Persistent custom prompt (e.g. `data/system_prompt.txt`)
    2. Versioned factory default prompt (`prompts/system_prompt.txt`)
    3. Hardcoded fallback string
    """
    # 1. Custom persistent override
    resolved_custom = Path(custom_path) if custom_path else resolve_custom_prompt_path()
    if resolved_custom.is_file():
        try:
            content = resolved_custom.read_text(encoding="utf-8").strip()
            if content:
                logger.info(
                    "Loaded custom system prompt from %s (%d chars)", resolved_custom, len(content)
                )
                return content
        except OSError as e:
            logger.warning("Failed to read custom prompt from %s: %s", resolved_custom, e)

    # 2. Versioned factory template
    resolved_base = Path(base_path) if base_path else DEFAULT_BASE_PROMPT_PATH
    if resolved_base.is_file():
        try:
            content = resolved_base.read_text(encoding="utf-8").strip()
            if content:
                return content
        except OSError as e:
            logger.warning("Failed to read base prompt from %s: %s", resolved_base, e)

    # 3. Emergency fallback
    return FALLBACK_SYSTEM_PROMPT


def save_system_prompt(
    new_prompt: str,
    custom_path: Path | str | None = None,
) -> Path:
    """
    Save a new custom system prompt to persistent storage.

    Persists to `data/system_prompt.txt` so it survives container rebuilds
    and restarts via Docker volume mounts.
    """
    cleaned = new_prompt.strip()
    if not cleaned:
        raise ValueError("System prompt cannot be empty.")

    target_path = Path(custom_path) if custom_path else resolve_custom_prompt_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(cleaned + "\n", encoding="utf-8")
    logger.info("Saved persistent system prompt to %s (%d chars)", target_path, len(cleaned))
    return target_path


def reset_system_prompt(
    custom_path: Path | str | None = None,
    base_path: Path | str | None = None,
) -> str:
    """
    Reset custom system prompt back to the factory template in `prompts/system_prompt.txt`.
    Deletes the persistent override file if present.
    """
    target_path = Path(custom_path) if custom_path else resolve_custom_prompt_path()
    if target_path.exists():
        try:
            target_path.unlink()
            logger.info("Removed custom system prompt at %s", target_path)
        except OSError as e:
            logger.error("Failed to delete custom prompt file at %s: %s", target_path, e)

    return load_system_prompt(custom_path=target_path, base_path=base_path)


def is_custom_prompt_active(custom_path: Path | str | None = None) -> bool:
    """Return True if a non-empty custom system prompt file exists."""
    target_path = Path(custom_path) if custom_path else resolve_custom_prompt_path()
    if target_path.is_file():
        try:
            return bool(target_path.read_text(encoding="utf-8").strip())
        except OSError:
            return False
    return False


# Re-export module-level active default system prompt
DEFAULT_SYSTEM_PROMPT = load_system_prompt()
