"""A small agent framework in the Agent / Task / Crew shape.

Deliberately built in-repo rather than pulled from CrewAI or AutoGen: both
frameworks default to hosted OpenAI models and routing them to a local Hugging
Face model means standing up an inference server alongside the app. The brief
asks for open-source models, so the orchestration is implemented directly
against the shared `LocalLLM` — the same roles, delegation and context passing,
with no API key and no extra process.

The pieces:

* `Agent`    — a role with a goal, a backstory, and optional tools.
* `Task`     — a unit of work assigned to an agent.
* `Crew`     — runs tasks in order, feeding each result to the next as context.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.llm.provider import get_llm

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """What an agent produced, plus how it got there."""

    agent: str
    output: str
    data: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    used_llm: bool = False

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "output": self.output,
            "data": self.data,
            "duration_seconds": round(self.duration_seconds, 3),
            "used_llm": self.used_llm,
        }


@dataclass
class Task:
    """A description of work plus the inputs the agent needs."""

    description: str
    agent: "Agent"
    inputs: dict = field(default_factory=dict)
    # Names of earlier tasks whose output should be passed in as context.
    context_from: list[str] = field(default_factory=list)
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.agent.name


class Agent:
    """A specialist role. Subclasses implement `execute`."""

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str = "",
        tools: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or {}

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        parts = [f"You are {self.role}.", f"Your goal: {self.goal}"]
        if self.backstory:
            parts.append(self.backstory)
        parts.append(
            "Work only from the data you are given. Never invent specifications "
            "or numbers. If a detail is absent, say it is not available."
        )
        return " ".join(parts)

    def think(self, prompt: str, max_new_tokens: int | None = None) -> tuple[str, bool]:
        """Ask the shared LLM. Returns `(text, used_llm)`.

        An empty string with `used_llm=False` tells the caller to fall back to
        its deterministic path, so a missing model degrades the prose rather
        than breaking the pipeline.
        """
        llm = get_llm()
        if llm is None:
            return "", False
        try:
            output = llm.complete(
                self.system_prompt(), prompt, max_new_tokens=max_new_tokens
            )
            return (output.strip(), True) if output.strip() else ("", False)
        except Exception as exc:
            logger.warning("%s: generation failed (%s)", self.name, exc)
            return "", False

    # ------------------------------------------------------------------
    def execute(self, task: Task, context: dict[str, AgentResult]) -> AgentResult:
        raise NotImplementedError

    def run(self, task: Task, context: dict[str, AgentResult] | None = None) -> AgentResult:
        """Execute a task with timing and error containment."""
        started = time.time()
        try:
            result = self.execute(task, context or {})
        except Exception as exc:
            logger.exception("%s failed", self.name)
            result = AgentResult(
                agent=self.name, output=f"{self.name} failed: {exc}", data={"error": str(exc)}
            )
        result.duration_seconds = time.time() - started
        return result

    def __repr__(self) -> str:
        return f"<Agent {self.name} ({self.role})>"


class Crew:
    """Runs a sequence of tasks, threading each result into the next."""

    def __init__(self, name: str, tasks: list[Task]):
        self.name = name
        self.tasks = tasks

    def kickoff(self) -> dict[str, AgentResult]:
        """Run every task in order and return `{task name: result}`."""
        results: dict[str, AgentResult] = {}

        for task in self.tasks:
            # Hand the agent only what it declared it needs, so a task cannot
            # accidentally depend on output it never asked for.
            context = {
                name: results[name] for name in task.context_from if name in results
            }
            missing = [n for n in task.context_from if n not in results]
            if missing:
                logger.warning(
                    "%s: context %s unavailable (upstream task did not run)",
                    task.name,
                    ", ".join(missing),
                )

            logger.info("[%s] %s -> %s", self.name, task.agent.name, task.description)
            results[task.name] = task.agent.run(task, context)

        return results

    def __repr__(self) -> str:
        return f"<Crew {self.name} tasks={len(self.tasks)}>"
