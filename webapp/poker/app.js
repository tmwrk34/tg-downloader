// Telegram WebApp Texas Hold'em Client
let tg = window.Telegram?.WebApp;
if (tg) {
    tg.ready();
    tg.expand();
}

let ws = null;
let currentTable = null;
let myUserId = null;

// Parse parameters from Telegram startapp or URL query
const urlParams = new URLSearchParams(window.location.search);
let chatId = urlParams.get("chat_id");
const startApp = tg?.initDataUnsafe?.start_param || "";
if (!chatId && startApp.startsWith("chat_")) {
    chatId = startApp.replace("chat_", "");
}
if (!chatId) {
    chatId = "-1002366722481"; // Default fallback table
}

const initData = tg?.initData || "";
if (tg?.initDataUnsafe?.user) {
    myUserId = tg.initDataUnsafe.user.id;
}

function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/poker?chat_id=${chatId}&init_data=${encodeURIComponent(initData)}`;

    const statusBadge = document.getElementById("connection-status");
    statusBadge.textContent = "З'єднання...";
    statusBadge.className = "status-badge";

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        statusBadge.textContent = "Підключено";
        statusBadge.className = "status-badge connected";
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === "state") {
                renderTableState(msg.data);
            } else if (msg.type === "notify") {
                showNotice(msg.message, msg.success);
            } else if (msg.type === "error") {
                showNotice(msg.message, false);
            }
        } catch (e) {
            console.error("WS Parse error", e);
        }
    };

    ws.onclose = () => {
        statusBadge.textContent = "Відключено";
        statusBadge.className = "status-badge error";
        setTimeout(initWebSocket, 3000);
    };

    ws.onerror = () => {
        statusBadge.textContent = "Помилка";
        statusBadge.className = "status-badge error";
    };
}

function showNotice(text, isSuccess) {
    const banner = document.getElementById("notice-banner");
    banner.textContent = text;
    banner.style.background = isSuccess ? "#27ae60" : "#c0392b";
    banner.classList.remove("hidden");
    if (tg?.HapticFeedback) {
        tg.HapticFeedback.notificationOccurred(isSuccess ? "success" : "error");
    }
    setTimeout(() => {
        banner.classList.add("hidden");
    }, 3500);
}

function renderTableState(data) {
    currentTable = data;

    // 1. Pot & Blinds
    document.getElementById("pot-amount").textContent = data.pot;
    document.getElementById("action-summary").textContent = data.last_action || "";

    // 2. Community Cards
    const ccContainer = document.getElementById("community-cards");
    ccContainer.innerHTML = "";
    for (let i = 0; i < 5; i++) {
        const cardStr = data.community_cards[i];
        const slot = document.createElement("div");
        if (cardStr) {
            slot.className = `card-slot ${isRedCard(cardStr) ? "red" : "black"}`;
            slot.textContent = cardStr;
        } else {
            slot.className = "card-slot empty";
        }
        ccContainer.appendChild(slot);
    }

    // 3. Render 6 Seats
    let iAmSeated = false;
    let myHoleCards = null;

    data.seats.forEach((seat, idx) => {
        const seatElem = document.getElementById(`seat-${idx}`);
        if (!seat) {
            seatElem.className = `seat seat-pos-0 empty seat-pos-${idx}`;
            seatElem.querySelector(".seat-avatar").textContent = "🪑";
            seatElem.querySelector(".seat-name").textContent = "Вільне місце";
            seatElem.querySelector(".seat-chips").textContent = "";
        } else {
            const isTurn = data.current_turn_seat === idx;
            seatElem.className = `seat seat-pos-${idx} ${isTurn ? "active-turn" : ""}`;
            seatElem.querySelector(".seat-avatar").textContent = seat.display_name.charAt(0).toUpperCase();
            seatElem.querySelector(".seat-name").textContent = seat.display_name;
            seatElem.querySelector(".seat-chips").textContent = `${seat.chips} 🪙`;

            if (seat.user_id === myUserId) {
                iAmSeated = true;
                myHoleCards = seat.hole_cards;
            }
        }
    });

    // 4. Render My Hole Cards
    const myCardsContainer = document.getElementById("my-cards");
    myCardsContainer.innerHTML = "";
    if (myHoleCards && myHoleCards.length === 2) {
        myHoleCards.forEach(c => {
            const cardElem = document.createElement("div");
            cardElem.className = `my-card ${isRedCard(c) ? "red" : "black"}`;
            cardElem.textContent = c;
            myCardsContainer.appendChild(cardElem);
        });
    } else {
        myCardsContainer.innerHTML = '<div class="card-placeholder">🂠</div><div class="card-placeholder">🂠</div>';
    }

    // 5. Action Buttons (My turn vs Lobby)
    const turnActions = document.getElementById("turn-actions");
    const lobbyActions = document.getElementById("lobby-actions");
    const va = data.viewer_actions;

    if (va && va.is_my_turn) {
        turnActions.classList.remove("hidden");
        lobbyActions.classList.add("hidden");

        document.getElementById("btn-check").style.display = va.can_check ? "inline-block" : "none";
        document.getElementById("btn-call").style.display = va.can_call ? "inline-block" : "none";
        if (va.can_call) {
            document.getElementById("btn-call").textContent = `Колл ${va.call_amount}`;
        }
        document.getElementById("btn-raise").style.display = va.can_raise ? "inline-block" : "none";

        // Update Raise Slider bounds
        const slider = document.getElementById("raise-slider");
        slider.min = va.min_raise;
        slider.max = va.max_raise;
        slider.value = va.min_raise;
        updateRaiseDisplay();

        if (tg?.HapticFeedback) {
            tg.HapticFeedback.impactOccurred("medium");
        }
    } else {
        turnActions.classList.add("hidden");
        document.getElementById("raise-panel").classList.add("hidden");
        lobbyActions.classList.remove("hidden");

        const btnStand = document.getElementById("btn-stand");
        btnStand.style.display = iAmSeated ? "inline-block" : "none";
    }
}

function isRedCard(cardStr) {
    return cardStr && (cardStr.includes("♥") || cardStr.includes("♦"));
}

function handleSeatClick(seatIndex) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action: "sit", seat: seatIndex }));
}

function sendAction(move, amount = null) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ action: "bet_action", move: move, amount: amount }));
}

function toggleRaiseSlider() {
    const rp = document.getElementById("raise-panel");
    rp.classList.toggle("hidden");
}

function updateRaiseDisplay() {
    const val = document.getElementById("raise-slider").value;
    document.getElementById("raise-val").textContent = val;
}

function setRaisePreset(preset) {
    if (!currentTable || !currentTable.viewer_actions) return;
    const va = currentTable.viewer_actions;
    const slider = document.getElementById("raise-slider");

    if (preset === "min") {
        slider.value = va.min_raise;
    } else if (preset === "half_pot") {
        slider.value = Math.min(va.max_raise, Math.max(va.min_raise, Math.floor(currentTable.pot / 2)));
    } else if (preset === "pot") {
        slider.value = Math.min(va.max_raise, Math.max(va.min_raise, currentTable.pot));
    } else if (preset === "allin") {
        slider.value = va.max_raise;
    }
    updateRaiseDisplay();
}

function sendRaise() {
    const val = parseInt(document.getElementById("raise-slider").value, 10);
    sendAction("raise", val);
    document.getElementById("raise-panel").classList.add("hidden");
}

function sendStartHand() {
    if (!ws) return;
    ws.send(JSON.stringify({ action: "start_hand" }));
}

function sendStand() {
    if (!ws) return;
    ws.send(JSON.stringify({ action: "stand" }));
}

function sendRebuy() {
    if (!ws) return;
    ws.send(JSON.stringify({ action: "rebuy" }));
}

window.addEventListener("DOMContentLoaded", initWebSocket);
