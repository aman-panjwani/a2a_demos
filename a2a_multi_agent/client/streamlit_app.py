import asyncio, inspect, json, os
from uuid import uuid4
from typing import Any, Dict, List, Union

import httpx
import streamlit as st
from a2a.client import ClientFactory, ClientConfig
from a2a.types import AgentCard, SendMessageRequest, MessageSendParams

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
ORCH_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:10002")
KNOWN_AGENT_URLS = [ORCH_URL]                 

# ────────────────────────────────────────────────────────────────────────────
# Discovery helpers
# ────────────────────────────────────────────────────────────────────────────
async def _load_card(session: httpx.AsyncClient, base: str) -> AgentCard | None:
    try:
        r = await session.get(f"{base}/.well-known/agent-card.json", timeout=5)
        r.raise_for_status()
        return AgentCard.model_validate(r.json())
    except Exception:
        return None

async def _load_all_cards() -> List[AgentCard]:
    async with httpx.AsyncClient() as s:
        cards = await asyncio.gather(*(_load_card(s, u) for u in KNOWN_AGENT_URLS))
    return [c for c in cards if c]

# ────────────────────────────────────────────────────────────────────────────
# Cache: event-loop, agent card, SDK client
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_runtime():
    loop = asyncio.new_event_loop()
    cards = loop.run_until_complete(_load_all_cards())
    factory = ClientFactory(ClientConfig())
    clients: Dict[str, Any] = {c.url: factory.create(c) for c in cards}
    return loop, cards, clients

loop, CARDS, CLIENTS = _get_runtime()
if not CARDS:
    st.error(f"Could not reach orchestrator at {ORCH_URL}")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="A2A Multi-Agent", layout="centered")
st.title("A2A Multi-Agent")

st.markdown(
    """
Ask a question&hellip; the orchestrator chooses the best helper agent for you.
"""
)

query = st.text_input("Your query", placeholder="e.g. Hello!")
submitted = st.button("Send", type="primary")

# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────
def _serialisable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if hasattr(obj, "dict"):
        return obj.dict(exclude_none=True)
    return obj

def _normalise(obj: Any) -> Any:
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return obj
    ser = _serialisable(obj)
    return ser if isinstance(ser, (dict, list)) else obj

def _artifact_text(art: Union[dict, Any]) -> str | None:
    parts = getattr(art, "parts", None) or art.get("parts", []) if isinstance(art, dict) else []
    for p in parts:
        kind = getattr(p, "kind", None) or p.get("kind")
        if kind == "text":
            return getattr(p, "text", None) or p.get("text")
    return getattr(art, "text", None) or art.get("text") if isinstance(art, dict) else None

def _walk(node: Any) -> str | None:
    if isinstance(node, dict):
        if "artifacts" in node:
            for art in node["artifacts"]:
                txt = _artifact_text(art)
                if txt:
                    return txt
        if "artifact" in node:
            txt = _artifact_text(node["artifact"])
            if txt:
                return txt
        if "status" in node and isinstance(node["status"], dict):
            msg = node["status"].get("message")
            if isinstance(msg, dict):
                txt = _walk(msg)
                if txt:
                    return txt
        for v in node.values():
            txt = _walk(v)
            if txt:
                return txt
    elif isinstance(node, list):
        for item in node:
            txt = _walk(item)
            if txt:
                return txt
    return None

def _extract_text(node: Any) -> str | None:
    return _walk(_normalise(node))

# ────────────────────────────────────────────────────────────────────────────
# Stream collector: stop at first event that contains text
# ────────────────────────────────────────────────────────────────────────────
async def _collect_stream(stream):
    last_event = None
    async for ev in stream:
        ev = ev[0] if isinstance(ev, tuple) else ev
        ev_dict = ev.model_dump(exclude_none=True) if hasattr(ev, "model_dump") else ev
        if _extract_text(ev_dict):
            return ev_dict                 
        last_event = ev_dict
    return last_event                       

def _send_sync(client, payload):
    try:
        maybe = client.send_message(payload)           
    except TypeError:
        wrapper = SendMessageRequest(
            id=uuid4().hex, params=MessageSendParams(message=payload)
        )
        maybe = client.send_message(wrapper)           

    if inspect.isasyncgen(maybe):
        return loop.run_until_complete(_collect_stream(maybe))
    if inspect.isawaitable(maybe):
        return loop.run_until_complete(maybe)
    return maybe

# ────────────────────────────────────────────────────────────────────────────
# On click
# ────────────────────────────────────────────────────────────────────────────
if submitted and query.strip():
    card = CARDS[0]                                 
    client = CLIENTS[card.url]

    st.info(f"**Using orchestrator:** {card.name}")

    user_msg = {
        "role": "user",
        "parts": [{"kind": "text", "text": query}],
        "messageId": uuid4().hex,
    }

    try:
        result = _send_sync(client, user_msg)
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

    answer = _extract_text(result) or json.dumps(_serialisable(result), indent=2)

    st.subheader("Answer")
    st.success(answer.replace("\n", "  \n"))

    with st.expander("Raw JSON response"):
        st.json(_serialisable(result))
