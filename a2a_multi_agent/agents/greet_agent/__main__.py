import click, os, sys, httpx, uvicorn
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from agents.greet_agent.quote_agent import GreetingQuoteAgent
from .agent_executor import GreetAgentExecutor

# Load environment variables from .env file (e.g., API keys, configs)
load_dotenv()

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10001) 
def main(host: str, port: int):
    """
    Entrypoint for running the Greeting Agent service.

    - Validates required env vars (e.g. GOOGLE_API_KEY).
    - Sets up the HTTP request handler and task store.
    - Starts the Starlette-based server using uvicorn.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set")
        sys.exit(1)

    # Create the request handler with agent executor and in-memory task store
    handler = DefaultRequestHandler(
        agent_executor=GreetAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # Build and configure the A2A Starlette application
    server = A2AStarletteApplication(
        agent_card=build_agent_card(host, port),
        http_handler=handler,
    )

    # Start the uvicorn server with specified host and port
    uvicorn.run(server.build(), host=host, port=port)

def build_agent_card(host: str, port: int) -> AgentCard:
    """
    Constructs the AgentCard metadata used to describe this agent to the A2A platform.

    This includes:
    - Name, description, version, and server URL
    - Supported input/output content types
    - Defined skills and capabilities (e.g., streaming, push notifications)
    """
    return AgentCard(
        name="Greeting Agent",
        description="Greets the user with a random inspirational quote.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, pushNotifications=True),
        defaultInputModes=GreetingQuoteAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=GreetingQuoteAgent.SUPPORTED_CONTENT_TYPES,
        skills=[
            AgentSkill(
                id="greet_with_quote",
                name="Greet with Quote",
                description="Returns a friendly greeting plus an inspirational quote.",
                tags=["greeting", "quote"],
                examples=["Hello!", "Say hi", "Greet me"],
            )
        ],
    )

# Run the CLI if this script is executed directly
if __name__ == "__main__":
    main()