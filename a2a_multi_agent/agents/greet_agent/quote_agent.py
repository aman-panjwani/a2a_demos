from typing import Any, AsyncIterable, Literal
from pydantic import BaseModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

class ResponseFormat(BaseModel):
    status: Literal["completed"] = "completed"
    message: str

class GreetingQuoteAgent:
    """
    Generates a friendly greeting + a short inspirational quote
    using Gemini 2.0-Flash. No tools, chains, or external APIs needed.
    """
    # Declares what content types the agent can handle
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self) -> None:
        # Initialize the Gemini Flash model with a bit of warmth (temperature=0.7)
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

        # System prompt that defines the agent's persona and response structure
        self.system_prompt = (
            "You are a warm, concise greeter. "
            "When the user greets you, reply with:\n"
            "1) A single-sentence greeting in the same language/tone.\n"
            "2) A newline.\n"
            "3) A short inspirational quote (max 25 words) followed by an en-dash "
            "and the author’s name, wrapped in double quotes.\n\n"
            "Example:\n"
            "Hello there!\n"
            "\"The journey of a thousand miles begins with one step. — Lao Tzu\""
        )

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Streams a two-step response:
        1. An initial progress update to simulate thinking.
        2. The final formatted greeting + quote.

        Args:
            query (str): The user’s greeting or message.
            session_id (str): A2A session context ID.

        Yields:
            Dicts with flags and generated content.
        """
        # Send a progress message while the model thinks
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Crafting your personalized greeting…",
        }

        # Construct system + user message stack
        messages = [
            ("system", self.system_prompt),
            ("user", query),
        ]

        # Send the request to Gemini and await the response
        result: AIMessage = await self.model.ainvoke(messages, RunnableConfig())

        # Clean up the final output and yield as task completion
        final_text = result.content.strip()

        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_text,
        }
