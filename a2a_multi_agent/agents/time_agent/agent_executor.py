from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

from .agent import TellTimeByLocationAgent as TellTimeAgent

class TellTimeAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.agent = TellTimeAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message:
            raise ValueError("No message provided")

        query = context.get_user_input()
        task = context.current_task

        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        async for step in self.agent.stream(query, task.context_id):
            if step["is_task_complete"]:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        taskId=task.id,
                        contextId=task.context_id,
                        artifact=new_text_artifact(
                            name="current_result",
                            description="Result of request to agent.",
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
                continue

            if step["require_user_input"]:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        taskId=task.id,
                        contextId=task.context_id,
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                step["content"], task.context_id, task.id
                            ),
                        ),
                        final=True,
                    )
                )
                continue

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

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel not supported")
