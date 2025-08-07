from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterable, Literal
import logging
import re
from zoneinfo import ZoneInfo, available_timezones

from pydantic import BaseModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)
memory = MemorySaver()

# --------------------------------------------------------------------------- #
# Tool: get_time                                                              #
# --------------------------------------------------------------------------- #
@tool
def get_time(location: str) -> dict[str, str]:
    """
    Return the current local time for *location*.

    *location* can be an exact IANA time-zone (``Europe/Paris``) **or** any
    substring that uniquely identifies one (``paris`` → ``Europe/Paris``).

    Returns
    -------
    dict
        ``{"current_time": "HH:MM:SS", "timezone": "Europe/Paris"}``
    """
    loc = location.strip().lower()

    if loc in map(str.lower, available_timezones()):
        tz = ZoneInfo(loc)
    else:
        match = next(
            (z for z in available_timezones()
             if re.search(rf"{re.escape(loc)}", z, re.I)),
            None,
        )
        if match:
            tz = ZoneInfo(match)
        else:
            raise ValueError(
                "Unknown location. Please provide a valid country or time-zone "
                "(e.g. 'Asia/Tokyo' or just 'tokyo')."
            )

    now = datetime.now(tz).strftime("%H:%M:%S")
    return {"current_time": now, "timezone": str(tz)}

# --------------------------------------------------------------------------- #
# Pydantic schema for the LLM’s structured response                           #
# --------------------------------------------------------------------------- #
class ResponseFormat(BaseModel):
    status: Literal["completed", "input_required", "error"]
    message: str

# --------------------------------------------------------------------------- #
# Main agent                                                                  #
# --------------------------------------------------------------------------- #
class TellTimeByLocationAgent:
    """
    ReAct-style agent (Gemini 2.0-Flash) that handles time-queries.
    """

    SYSTEM_INSTRUCTION = (
        "You are a specialised assistant for time-related queries. "
        "When the user asks the time **and** specifies a place (e.g. "
        "'What is the time in Tokyo?'), call the `get_time` tool with that "
        "location. If the user does **not** specify a location, ask them "
        "which place they mean. "
        "Always format your final answer exactly like:\n"
        "  It is HH:MM:SS in <city / country>."
    )

    RESPONSE_FORMAT_INSTRUCTION = (
        "Return JSON with keys `status` and `message`.\n"
        "• Use `completed` when you have the final answer.\n"
        "• Use `input_required` when you need a location.\n"
        "• Use `error` only on failure."
    )

    # Declares the MIME types this agent supports in A2A
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self) -> None:
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = [get_time]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

    # --------------------------------------------------------------------- #
    # A2A-style async stream interface                                       #
    # --------------------------------------------------------------------- #
    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """Yield progress updates and finally the structured result."""
        cfg: RunnableConfig = {"configurable": {"thread_id": session_id}}
        inputs = {"messages": [("user", query)]}

        for step in self.graph.stream(inputs, cfg, stream_mode="values"):
            msg = step["messages"][-1]

            # The agent is about to call the tool
            if isinstance(msg, AIMessage) and msg.tool_calls:
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Fetching the local time…",
                }

            # The tool has returned; the agent is formatting
            elif isinstance(msg, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Formatting the result…",
                }

        # After the stream ends send the final result
        yield self._final(cfg)

    # --------------------------------------------------------------------- #
    # Helpers                                                                #
    # --------------------------------------------------------------------- #
    def _final(self, cfg: RunnableConfig) -> dict[str, Any]:
        state = self.graph.get_state(cfg)
        structured = state.values.get("structured_response")

        if isinstance(structured, ResponseFormat):
            done = structured.status == "completed"
            return {
                "is_task_complete": done,
                "require_user_input": not done,
                "content": structured.message,
            }

        logger.warning("Structured response missing or malformed: %s", structured)
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "Sorry, I couldn’t get the time. Please try again?",
        }