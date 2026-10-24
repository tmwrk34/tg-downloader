import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    bot_token: str
    allowed_chat_ids: set[int]
    admin_user_ids: set[int]
    allowed_group_ids: set[int]
    local_bot_api_url: str | None = None

    @property
    def max_payload_bytes(self) -> int:
        """
        Return the maximum allowable upload payload size in bytes.
        50MB for default cloud Telegram Bot API, 2000MB (2GB) for local Bot API server.
        """
        if self.local_bot_api_url:
            return 2000 * 1024 * 1024
        return 50 * 1024 * 1024


def parse_chat_ids(raw_ids: str | None) -> tuple[set[int], set[int], set[int]]:
    """
    Parse a comma-separated string of Telegram chat IDs and separate them into:
    - all_chat_ids: complete set for O(1) whitelist lookup
    - admin_user_ids: positive IDs (> 0) representing private personal/admin chats
    - allowed_group_ids: negative IDs (< 0) representing group/supergroup chats
    """
    if not raw_ids or not raw_ids.strip():
        raise ValueError("ALLOWED_CHAT_IDS not found or empty in environment")

    all_ids: set[int] = set()
    admin_ids: set[int] = set()
    group_ids: set[int] = set()

    for item in raw_ids.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        try:
            chat_id = int(cleaned)
        except ValueError as err:
            raise ValueError(f"Invalid chat_id in ALLOWED_CHAT_IDS: '{cleaned}'") from err

        all_ids.add(chat_id)
        if chat_id > 0:
            admin_ids.add(chat_id)
        elif chat_id < 0:
            group_ids.add(chat_id)

    if not all_ids:
        raise ValueError("ALLOWED_CHAT_IDS does not contain any valid chat IDs")

    return all_ids, admin_ids, group_ids


def load_config() -> AppConfig:
    token = os.getenv("BOT_TOKEN")
    if not token or not token.strip():
        raise ValueError("BOT_TOKEN not found in environment variables")

    raw_chat_ids = os.getenv("ALLOWED_CHAT_IDS")
    all_ids, admin_ids, group_ids = parse_chat_ids(raw_chat_ids)

    local_api_url = os.getenv("LOCAL_BOT_API_URL")
    if local_api_url:
        local_api_url = local_api_url.strip()
        if not local_api_url:
            local_api_url = None

    return AppConfig(
        bot_token=token.strip(),
        allowed_chat_ids=all_ids,
        admin_user_ids=admin_ids,
        allowed_group_ids=group_ids,
        local_bot_api_url=local_api_url,
    )


# Global configuration instance
config = load_config() if os.getenv("BOT_TOKEN") and os.getenv("ALLOWED_CHAT_IDS") else None
