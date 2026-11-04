"""
aiohttp Web & WebSocket server for Telegram Mini Apps (TWA) Tavern Games.
Serves static WebApp files and streams real-time poker game state updates over WebSockets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from services.games.auth import validate_telegram_init_data
from services.games.manager import games_manager
from services.games.poker.engine import PokerTable

logger = logging.getLogger(__name__)

STATIC_WEBAPP_DIR = Path(__file__).resolve().parent.parent.parent / "webapp"


class GamesWebServer:
    def __init__(self, bot_token: str, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.bot_token = bot_token
        self.host = host
        self.port = port
        self.app = web.Application()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

        # chat_id -> set of active WebSocket connections
        self._table_connections: dict[int, set[web.WebSocketResponse]] = {}
        # ws -> user_id
        self._ws_user_map: dict[web.WebSocketResponse, int] = {}

        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.router.add_get("/api/health", self._handle_health)
        self.app.router.add_get("/ws/poker", self._handle_poker_ws)

        # Serve static WebApp files
        poker_webapp_dir = STATIC_WEBAPP_DIR / "poker"
        poker_webapp_dir.mkdir(parents=True, exist_ok=True)
        self.app.router.add_static("/poker", poker_webapp_dir, show_index=True)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "tavern-games-server"})

    async def _handle_poker_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=25.0)
        await ws.prepare(request)

        chat_id_raw = request.query.get("chat_id")
        init_data_raw = request.query.get("init_data") or request.headers.get(
            "X-Telegram-Init-Data", ""
        )

        try:
            chat_id = int(chat_id_raw or 0)
        except ValueError:
            await ws.send_json({"type": "error", "message": "Невірний chat_id!"})
            await ws.close()
            return ws

        validated_user = validate_telegram_init_data(init_data_raw, self.bot_token)
        if not validated_user:
            await ws.send_json({"type": "error", "message": "Помилка авторизації Telegram WebApp!"})
            await ws.close()
            return ws

        user_id = validated_user.user_id
        display_name = validated_user.first_name
        username = validated_user.username

        # Register connection
        if chat_id not in self._table_connections:
            self._table_connections[chat_id] = set()
        self._table_connections[chat_id].add(ws)
        self._ws_user_map[ws] = user_id

        table = games_manager.get_or_create_poker_table(chat_id)
        # Wire table state change broadcast
        table._on_state_change = self._broadcast_table_state

        # Send initial table state to viewer
        await ws.send_json(
            {
                "type": "state",
                "data": table.to_view_dict(viewer_user_id=user_id),
            }
        )

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._process_ws_command(
                            chat_id, user_id, display_name, username, table, data, ws
                        )
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "message": "Невалідний JSON"})
                elif msg.type == WSMsgType.ERROR:
                    logger.debug("WS error: %s", ws.exception())
        finally:
            # Unregister connection
            if chat_id in self._table_connections:
                self._table_connections[chat_id].discard(ws)
                if not self._table_connections[chat_id]:
                    del self._table_connections[chat_id]
            self._ws_user_map.pop(ws, None)

        return ws

    async def _process_ws_command(
        self,
        chat_id: int,
        user_id: int,
        display_name: str,
        username: str | None,
        table: PokerTable,
        data: dict[str, Any],
        ws: web.WebSocketResponse,
    ) -> None:
        action = data.get("action")

        if action == "sit":
            seat = data.get("seat")
            ok, msg = table.sit_down(
                user_id=user_id, display_name=display_name, username=username, preferred_seat=seat
            )
            await ws.send_json({"type": "notify", "success": ok, "message": msg})
            await self._broadcast_table_state(table)

        elif action == "stand":
            ok, msg = table.stand_up(user_id=user_id)
            await ws.send_json({"type": "notify", "success": ok, "message": msg})
            await self._broadcast_table_state(table)

        elif action == "start_hand":
            ok, msg = table.start_hand()
            await ws.send_json({"type": "notify", "success": ok, "message": msg})
            await self._broadcast_table_state(table)

        elif action == "bet_action":
            move_str = data.get("move", "")
            amount = data.get("amount")
            ok, msg = table.player_move(user_id, move_str, raise_amount=amount)
            await ws.send_json({"type": "notify", "success": ok, "message": msg})
            await self._broadcast_table_state(table)

        elif action == "rebuy":
            ok, msg = table.rebuy(user_id=user_id)
            await ws.send_json({"type": "notify", "success": ok, "message": msg})
            await self._broadcast_table_state(table)

    async def _broadcast_table_state(self, table: PokerTable) -> None:
        """Broadcast state updates to all active WebApp clients with their individual hole cards."""
        chat_id = table.chat_id
        connections = self._table_connections.get(chat_id, set())
        if not connections:
            return

        dead_connections = set()
        for ws in connections:
            if ws.closed:
                dead_connections.add(ws)
                continue
            viewer_uid = self._ws_user_map.get(ws)
            state_dict = table.to_view_dict(viewer_user_id=viewer_uid)
            try:
                await ws.send_json({"type": "state", "data": state_dict})
            except Exception as e:
                logger.debug("Failed sending WS update: %s", e)
                dead_connections.add(ws)

        for dead_ws in dead_connections:
            connections.discard(dead_ws)
            self._ws_user_map.pop(dead_ws, None)

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info("Tavern Games WebApp Server running on http://%s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            logger.info("Tavern Games WebApp Server stopped.")
