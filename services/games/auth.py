"""
Telegram WebApp initData HMAC-SHA256 signature verification.
Follows official Telegram specification:
https://core.telegram.org/bots/webapps#validating-data-received-via-the-web-app
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import urllib.parse
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidatedWebUser:
    user_id: int
    first_name: str
    last_name: str | None = None
    username: str | None = None
    language_code: str | None = None


def validate_telegram_init_data(init_data_raw: str, bot_token: str) -> ValidatedWebUser | None:
    """
    Validate Telegram WebApp initData string using bot token secret key.
    Returns ValidatedWebUser on success or None on signature mismatch/tampering.
    """
    if not init_data_raw or not init_data_raw.strip():
        # Allow dev mode bypass if explicitly configured
        dev_mode = os.getenv("WEBAPP_DEV_MODE", "0") in ("1", "true", "yes")
        if dev_mode:
            logger.warning("WEBAPP_DEV_MODE enabled: granting mock test user")
            return ValidatedWebUser(user_id=999999, first_name="DevPlayer", username="dev_tester")
        return None

    try:
        parsed = urllib.parse.parse_qsl(init_data_raw, keep_blank_values=True)
        data_dict = dict(parsed)

        received_hash = data_dict.pop("hash", None)
        if not received_hash:
            return None

        # Build data_check_string in alphabetical order of keys
        sorted_items = sorted(data_dict.items(), key=lambda x: x[0])
        data_check_string = "\n".join(f"{k}={v}" for k, v in sorted_items)

        # 1. secret_key = HMAC_SHA256("WebAppData", bot_token)
        secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()

        # 2. calculated_hash = HMAC_SHA256(secret_key, data_check_string)
        calculated_hash = hmac.new(
            secret_key, data_check_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(calculated_hash, received_hash):
            logger.warning("Telegram WebApp initData signature mismatch!")
            return None

        user_raw = data_dict.get("user")
        if not user_raw:
            return None

        user_obj: dict[str, Any] = json.loads(user_raw)
        return ValidatedWebUser(
            user_id=int(user_obj["id"]),
            first_name=str(user_obj.get("first_name") or "Traveler"),
            last_name=user_obj.get("last_name"),
            username=user_obj.get("username"),
            language_code=user_obj.get("language_code"),
        )
    except Exception as e:
        logger.error("Error validating initData: %s", e)
        return None
