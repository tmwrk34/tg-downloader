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
    max_concurrent_downloads: int = 2
    max_concurrent_transcodes: int = 1
    munito_api_key: str | None = None
    munito_model: str = "deepseek-ai/deepseek-v4.1-flash-1017"
    munito_base_url: str = "https://api.munito.ai/inference/v1"
    munito_system_prompt: str | None = None

    @property
    def max_payload_bytes(self) -> int:
        """
        Return the maximum allowable upload payload size in bytes.
        50MB for default cloud Telegram Bot API, 2000MB (2GB) for local Bot API server.
        """
        if self.local_bot_api_url:
            return 2000 * 1024 * 1024
        return 50 * 1024 * 1024


def _parse_positive_int(env_key: str, default: int) -> int:
    raw = os.getenv(env_key)
    if not raw or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError as err:
        raise ValueError(f"Invalid integer for {env_key}: '{raw}'") from err
    if val < 1:
        raise ValueError(f"{env_key} must be >= 1, got {val}")
    return val


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
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    if local_api_url and local_api_url.strip():
        local_api_url = local_api_url.strip()
        # In docker environment, rewrite localhost/127.0.0.1 to telegram-bot-api service
        if "localhost" in local_api_url or "127.0.0.1" in local_api_url:
            local_api_url = local_api_url.replace("localhost", "telegram-bot-api").replace(
                "127.0.0.1", "telegram-bot-api"
            )
    elif api_id and api_hash and api_id.strip() and api_hash.strip():
        # Auto-activate local Bot API container when credentials are present
        local_api_url = "http://telegram-bot-api:8081"
    else:
        local_api_url = None

    max_concurrent_downloads = _parse_positive_int("MAX_CONCURRENT_DOWNLOADS", 2)
    max_concurrent_transcodes = _parse_positive_int("MAX_CONCURRENT_TRANSCODES", 1)

    munito_api_key = os.getenv("MUNITO_API_KEY")
    munito_api_key = munito_api_key.strip() if munito_api_key and munito_api_key.strip() else None
    munito_model = os.getenv("MUNITO_MODEL", "deepseek-ai/deepseek-v4.1-flash-1017").strip()
    munito_base_url = os.getenv("MUNITO_BASE_URL", "https://api.munito.ai/inference/v1").strip()
    munito_system_prompt = os.getenv("MUNITO_SYSTEM_PROMPT")
    munito_system_prompt = (
        munito_system_prompt.strip()
        if munito_system_prompt and munito_system_prompt.strip()
        else None
    )

    return AppConfig(
        bot_token=token.strip(),
        allowed_chat_ids=all_ids,
        admin_user_ids=admin_ids,
        allowed_group_ids=group_ids,
        local_bot_api_url=local_api_url,
        max_concurrent_downloads=max_concurrent_downloads,
        max_concurrent_transcodes=max_concurrent_transcodes,
        munito_api_key=munito_api_key,
        munito_model=munito_model,
        munito_base_url=munito_base_url,
        munito_system_prompt=munito_system_prompt,
    )
