from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

from .quote_agent import GreetingQuoteAgent 

class GreetAgentExecutor(AgentExecutor):
    """Bridges GreetingQuoteAgent ↔ A2A runtime."""

    def __init__(self) -> None:
        # Initialize the executor with an instance of our custom agent
        self.agent = GreetingQuoteAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Handles the main agent execution lifecycle.

        - Validates user input
        - Starts a new task if needed
        - Streams response from the agent
        - Pushes progress and completion events to the event queue
        """
        if not context.message:
            raise ValueError("No message provided")

        # Extract user query from the message context
        query = context.get_user_input()

        # Either use the current task or create a new one
        task = context.current_task or new_task(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        # Stream steps from the agent one-by-one
        async for step in self.agent.stream(query, task.context_id):
            if step["is_task_complete"]:
                # Task is finished — send final artifact and mark it complete
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        taskId=task.id,
                        contextId=task.context_id,
                        artifact=new_text_artifact(
                            name="greeting",
                            description="Greeting with quote",
                            text=step["content"],
                        ),
                        append=False,
                        lastChunk=True,
                    )
                )
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        taskId=task.id,
                        contextId=task.context_id,
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                    )
                )
            else: 
                # Task is in progress — update the client with streaming content
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        taskId=task.id,
                        contextId=task.context_id,
                        status=TaskStatus(
                            state=TaskState.working,
                            message=new_agent_text_message(
                                step["content"], task.context_id, task.id
                            ),
                        ),
                        final=False,
                    )
                )

    async def cancel(self, *_): 
        """
        Cancelling execution isn't currently supported.
        """
        raise NotImplementedError
