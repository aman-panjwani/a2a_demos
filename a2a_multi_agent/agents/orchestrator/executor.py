from __future__ import annotations

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

class OrchestratorExecutor(AgentExecutor):
    """Relays helper-agent events and appends one final completed status."""

    def __init__(self, router):
        self.router = router

    async def execute(self, ctx: RequestContext, q: EventQueue) -> None:
        if not ctx.message:
            raise ValueError("No message provided")

        query     = ctx.get_user_input()
        orch_task = ctx.current_task or new_task(ctx.message)
        if ctx.current_task is None:
            await q.enqueue_event(orch_task)

        async for helper_event in self.router.stream(query, orch_task.context_id):

            ev = (
                helper_event.model_dump(exclude_none=True)
                if hasattr(helper_event, "model_dump")
                else helper_event
            )

            if "artifact" in ev or "artifacts" in ev:
                artev = _make_artifact_event(ev, orch_task)
                await q.enqueue_event(artev)

                answer = _first_text(artev.artifact.model_dump())
                await _enqueue_completed(q, orch_task, answer)
                break

            if ev.get("kind") == "task" and "status" in ev:
                stat = _make_status_event(ev, orch_task)
                await q.enqueue_event(stat)
                continue

            if ev.get("is_task_complete"):
                answer = ev["content"]
                await q.enqueue_event(
                    TaskArtifactUpdateEvent(
                        taskId=orch_task.id,
                        contextId=orch_task.context_id,
                        artifact=new_text_artifact(
                            name="answer",
                            description="Final answer from downstream agent",
                            text=answer,
                        ),
                        append=False,
                        lastChunk=True,
                    )
                )
                await _enqueue_completed(q, orch_task, answer)
                break

            await q.enqueue_event(
                TaskStatusUpdateEvent(
                    taskId=orch_task.id,
                    contextId=orch_task.context_id,
                    status=TaskStatus(
                        state=TaskState.working,
                        message=new_agent_text_message(
                            ev["content"], orch_task.context_id, orch_task.id
                        ),
                    ),
                    final=False,
                )
            )

    async def cancel(self, *_):
        raise NotImplementedError("Cancel not supported")


def _make_artifact_event(d: dict, task) -> TaskArtifactUpdateEvent:
    art = d["artifact"] if "artifact" in d else d["artifacts"][0]
    return TaskArtifactUpdateEvent(
        taskId=task.id, contextId=task.context_id,
        artifact=art, append=False, lastChunk=True
    )


def _make_status_event(d: dict, task) -> TaskStatusUpdateEvent:
    return TaskStatusUpdateEvent(
        taskId=task.id,
        contextId=task.context_id,
        status=TaskStatus.model_validate(d["status"]),
        final=False,
    )


def _first_text(node: dict | list) -> str | None:
    """DFS for parts[*].text."""
    if isinstance(node, dict):
        if node.get("kind") == "text":
            return node.get("text")
        for v in node.values():
            t = _first_text(v)
            if t:
                return t
    elif isinstance(node, list):
        for item in node:
            t = _first_text(item)
            if t:
                return t
    return None


async def _enqueue_completed(q: EventQueue, task, text: str):
    """Send the final completed status that Streamlit will display."""
    await q.enqueue_event(
        TaskStatusUpdateEvent(
            taskId=task.id,
            contextId=task.context_id,
            status=TaskStatus(
                state=TaskState.completed,
                message=new_agent_text_message(text, task.context_id, task.id),
            ),
            final=True,
        )
    )
