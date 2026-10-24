import pytest


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Ensure baseline bot configuration is present across all tests."""
    monkeypatch.setenv("BOT_TOKEN", "123456:dummy_test_token_for_pytest")
    monkeypatch.setenv("ALLOWED_CHAT_IDS", "123456789,-100123456789")
