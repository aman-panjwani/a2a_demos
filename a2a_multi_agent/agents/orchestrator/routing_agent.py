from __future__ import annotations

import asyncio, httpx, json, logging, os, re
from typing import Any, AsyncIterable, Dict, List
from uuid import uuid4

from a2a.client import ClientFactory, ClientConfig
from a2a.types import AgentCard
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

from httpx import ReadError, RemoteProtocolError
from asyncio import CancelledError

logger = logging.getLogger(__name__)
DISCOVERY_INTERVAL = 600      

async def _fetch_card(session: httpx.AsyncClient, url: str) -> AgentCard | None:
    try:
        res = await session.get(f"{url}/.well-known/agent-card.json", timeout=5)
        res.raise_for_status()
        return AgentCard.model_validate(res.json())
    except Exception as exc:
        logger.warning("Failed to get card from %s: %s", url, exc)
        return None


class OrchestratorRoutingAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    SYSTEM_PROMPT = (
        "You are an orchestrator deciding which helper agent should answer the "
        "user. You will get:\n"
        "  • a JSON list of agents (url, name, skills)\n"
        "  • the user question\n\n"
        "Reply with ONLY the chosen agent's URL, or the single word NONE."
    )

    def __init__(self, peer_urls: List[str]):
        self.peer_urls = peer_urls
        self.cards: Dict[str, AgentCard] = {}
        self.last_discovery = 0.0

        model = os.getenv("ORCH_MODEL", "gemini-1.5-flash")
        self.model = ChatGoogleGenerativeAI(model=model)
        self.client_factory = ClientFactory(ClientConfig())

        asyncio.get_event_loop().run_until_complete(self._discover())

    async def _discover(self):
        async with httpx.AsyncClient() as s:
            found = await asyncio.gather(*(_fetch_card(s, u) for u in self.peer_urls))
        self.cards = {c.url: c for c in found if c}
        self.last_discovery = asyncio.get_event_loop().time()
        logger.info("Discovered %d helper agents", len(self.cards))

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[Dict[str, Any]]:

        if asyncio.get_event_loop().time() - self.last_discovery > DISCOVERY_INTERVAL:
            await self._discover()

        helper_url = await self._choose_agent(query)
        if helper_url == "NONE" or helper_url not in self.cards:
            helper_url = self._fallback_by_tags(query.lower())

        if helper_url == "NONE" or helper_url not in self.cards:
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "I’m sorry – none of my helper agents can answer that.",
            }
            return

        card   = self.cards[helper_url]
        client = self.client_factory.create(card)

        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": f"Asking **{card.name}**…",
        }

        user_msg = {
            "role": "user",
            "parts": [{"kind": "text", "text": query}],
            "messageId": uuid4().hex,
        }

        stream = client.send_message(user_msg)

        try:                                   
            async for ev in stream:
                yield ev[0] if isinstance(ev, tuple) else ev
        except (CancelledError, ReadError, RemoteProtocolError) as exc:
            logger.debug("Down-stream SSE closed: %s", exc)

    async def _choose_agent(self, question: str) -> str:
        agents_json = json.dumps(
            [
                {
                    "url": c.url,
                    "name": c.name,
                    "description": c.description,
                    "skills": [
                        {
                            "id": s.id,
                            "name": s.name,
                            "description": s.description,
                            "tags": s.tags,
                            "examples": s.examples,
                        }
                        for s in c.skills or []
                    ],
                }
                for c in self.cards.values()
            ],
            indent=2,
        )

        prompt = [
            ("system", self.SYSTEM_PROMPT),
            ("user",
             f"Agents list:\n{agents_json}\n\n"
             f"User question:\n{question}\n\nChosen URL:")
        ]
        reply: AIMessage = await self.model.ainvoke(prompt)
        match = re.search(r"https?://[^\s\"'<>]+", reply.content)
        url   = match.group(0) if match else "NONE"
        logger.info("Gemini routing decision: %s", url)
        return url

    def _fallback_by_tags(self, q: str) -> str:
        for c in self.cards.values():
            for sk in c.skills:
                if sk.id and sk.id.lower() in q:
                    return c.url
                if any(tag.lower() in q for tag in sk.tags or []):
                    return c.url
        return "NONE"
