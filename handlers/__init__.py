from __future__ import annotations

from aiogram import Router

from handlers.ai import ai_router
from handlers.base import base_router
from handlers.chatter import chatter_router
from handlers.dice import dice_router
from handlers.downloader import downloader_router
from handlers.games import games_router
from handlers.quote import quote_router

# Order matters for message interceptors:
# 1. Base commands (/start, /help, /ping)
# 2. Medieval quote generator (/q, /quote, /цитата)
# 3. Dice roller (/roll, inline query, natural language rolls)
# 4. Tavern Games Hub & Poker (/games, /poker)
# 5. Chatter admin commands & callbacks (/chatter)
# 6. Downloader (MediaUrlFilter intercepts TikTok, Pinterest, YouTube, Instagram)
# 7. AI commands & AI/Chatter fallback listener (handles @bot mentions and Markov learning)
router = Router(name="main_router")
main_router = router
router.include_routers(
    base_router,
    quote_router,
    dice_router,
    games_router,
    chatter_router,
    downloader_router,
    ai_router,
)

__all__ = [
    "ai_router",
    "base_router",
    "chatter_router",
    "dice_router",
    "downloader_router",
    "games_router",
    "main_router",
    "quote_router",
    "router",
]
