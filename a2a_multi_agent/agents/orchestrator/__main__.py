import asyncio, os, sys, uvicorn, httpx, json
from dotenv import load_dotenv
import click

from .routing_agent import OrchestratorRoutingAgent
from .executor import OrchestratorExecutor

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

load_dotenv()

@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10002)
@click.option(
    "--peers",
    help="Comma-separated list of base URLs to discover (e.g. http://localhost:10000)",
    envvar="A2A_PEERS",
)
def main(host: str, port: int, peers: str):
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set"); sys.exit(1)

    peer_urls = [p.strip() for p in peers.split(",") if p.strip()]
    if not peer_urls:
        print("No peer agent URLs supplied (use --peers or A2A_PEERS)"); sys.exit(1)

    router = OrchestratorRoutingAgent(peer_urls)

    handler = DefaultRequestHandler(
        agent_executor=OrchestratorExecutor(router),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=build_agent_card(host, port),
        http_handler=handler,
    )
    uvicorn.run(server.build(), host=host, port=port)

def build_agent_card(host: str, port: int) -> AgentCard:
    return AgentCard(
        name="Orchestrator Agent",
        description="Selects the best downstream agent to answer the user query.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True, pushNotifications=True),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="auto_route",
                name="Automatic routing",
                description="Figures out which helper agent is most suitable.",
                tags=["routing"],        
                examples=[]
            )
        ],
    )


if __name__ == "__main__":
    main()
