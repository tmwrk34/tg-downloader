import os
from unittest.mock import patch

import pytest
from aiogram.client.session.aiohttp import AiohttpSession

from config import AppConfig, load_config


def test_app_config_max_payload_bytes():
    # Cloud mode (default)
    cfg_cloud = AppConfig(
        bot_token="test_token",
        allowed_chat_ids={123},
        admin_user_ids={123},
        allowed_group_ids=set(),
        local_bot_api_url=None,
    )
    assert cfg_cloud.max_payload_bytes == 50 * 1024 * 1024

    # Local Bot API mode (2GB)
    cfg_local = AppConfig(
        bot_token="test_token",
        allowed_chat_ids={123},
        admin_user_ids={123},
        allowed_group_ids=set(),
        local_bot_api_url="http://telegram-bot-api:8081",
    )
    assert cfg_local.max_payload_bytes == 2000 * 1024 * 1024


def test_load_config_with_local_bot_api():
    env_mock = {
        "BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "ALLOWED_CHAT_IDS": "12345,-100123456",
        "LOCAL_BOT_API_URL": "http://telegram-bot-api:8081",
    }
    with patch.dict(os.environ, env_mock, clear=True):
        cfg = load_config()
        assert cfg.local_bot_api_url == "http://telegram-bot-api:8081"
        assert cfg.max_payload_bytes == 2000 * 1024 * 1024


def test_load_config_without_local_bot_api():
    env_mock = {
        "BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "ALLOWED_CHAT_IDS": "12345",
    }
    with patch.dict(os.environ, env_mock, clear=True):
        cfg = load_config()
        assert cfg.local_bot_api_url is None
        assert cfg.max_payload_bytes == 50 * 1024 * 1024


def test_load_config_auto_activate_with_api_credentials():
    env_mock = {
        "BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "ALLOWED_CHAT_IDS": "12345",
        "TELEGRAM_API_ID": "123456",
        "TELEGRAM_API_HASH": "abcdef123456",
    }
    with patch.dict(os.environ, env_mock, clear=True):
        cfg = load_config()
        assert cfg.local_bot_api_url == "http://telegram-bot-api:8081"
        assert cfg.max_payload_bytes == 2000 * 1024 * 1024


def test_load_config_rewrite_localhost():
    env_mock = {
        "BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "ALLOWED_CHAT_IDS": "12345",
        "LOCAL_BOT_API_URL": "http://localhost:8081",
    }
    with patch.dict(os.environ, env_mock, clear=True):
        cfg = load_config()
        assert cfg.local_bot_api_url == "http://telegram-bot-api:8081"
        assert cfg.max_payload_bytes == 2000 * 1024 * 1024


def test_local_bot_api_session_creation():
    from aiogram.client.telegram import TelegramAPIServer

    server = TelegramAPIServer.from_base("http://telegram-bot-api:8081", is_local=False)
    session = AiohttpSession(api=server)
    assert session.api.base == "http://telegram-bot-api:8081/bot{token}/{method}"
    assert session.api.file == "http://telegram-bot-api:8081/file/bot{token}/{path}"


def test_load_config_worker_concurrency_limits():
    env_mock = {
        "BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "ALLOWED_CHAT_IDS": "12345",
        "MAX_CONCURRENT_DOWNLOADS": "4",
        "MAX_CONCURRENT_TRANSCODES": "2",
    }
    with patch.dict(os.environ, env_mock, clear=True):
        cfg = load_config()
        assert cfg.max_concurrent_downloads == 4
        assert cfg.max_concurrent_transcodes == 2


def test_load_config_worker_concurrency_defaults_and_validation():
    env_defaults = {
        "BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "ALLOWED_CHAT_IDS": "12345",
    }
    with patch.dict(os.environ, env_defaults, clear=True):
        cfg = load_config()
        assert cfg.max_concurrent_downloads == 2
        assert cfg.max_concurrent_transcodes == 1

    env_zero = {
        **env_defaults,
        "MAX_CONCURRENT_DOWNLOADS": "0",
    }
    with patch.dict(os.environ, env_zero, clear=True), pytest.raises(ValueError):
        load_config()

    env_invalid = {
        **env_defaults,
        "MAX_CONCURRENT_TRANSCODES": "invalid",
    }
    with patch.dict(os.environ, env_invalid, clear=True), pytest.raises(ValueError):
        load_config()
