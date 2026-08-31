"""Pre-built crews: the workflows the API and CLI expose.

Two orchestration backends sit behind one interface:

* **crewai** — real CrewAI `Agent`/`Task`/`Crew` objects running on the local
  model (`crewai_backend.py`). This is the default when CrewAI is installed.
* **native** — the built-in orchestrator in `base.py`, used when CrewAI is not
  installed or no LLM can be loaded. It also has template fallbacks, so the
  system keeps answering on hardware that cannot run a generative model.

`AGENT_FRAMEWORK` in `.env` forces one or the other; `auto` (the default) picks
CrewAI when it is usable.

In both backends the data agent runs first and the writing agents consume its
output. That ordering is the point of the split: the reviewer has no database
access, so it has nothing to work from except retrieved facts.
"""

from __future__ import annotations

import logging
import time

import config
from src.database.db import session_scope
from src.database.repository import extract_mentioned_phones

from . import crewai_backend
from .base import AgentResult, Crew, Task
from .specialists import ComparisonAgent, ReviewAgent, SpecificationAgent

logger = logging.getLogger(__name__)


def _serialise(results: dict[str, AgentResult]) -> dict:
    return {name: result.to_dict() for name, result in results.items()}


def active_backend() -> str:
    """Which orchestrator will run: `"crewai"` or `"native"`."""
    requested = config.AGENT_FRAMEWORK

    if requested == "native":
        return "native"

    if crewai_backend.is_available():
        return "crewai"

    if requested == "crewai":
        # Explicitly requested but unusable — say why rather than silently
        # substituting a different implementation.
        logger.warning(
            "AGENT_FRAMEWORK=crewai but CrewAI is unavailable (%s); "
            "falling back to the native orchestrator",
            crewai_backend.status(),
        )
    return "native"


def backend_status() -> dict:
    return {"active": active_backend(), **crewai_backend.status()}


def _crew_agent_trace(roles: list[str], seconds: float) -> dict:
    """Report CrewAI agents in the same shape the native backend uses."""
    return {
        f"crewai_task_{i}": {
            "agent": role,
            "output": "",
            "data": {"framework": "crewai"},
            # CrewAI does not expose per-task timings, so the crew's total is
            # divided evenly rather than reported as if it were measured.
            "duration_seconds": round(seconds / max(len(roles), 1), 3),
            "used_llm": True,
        }
        for i, role in enumerate(roles, start=1)
    }


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------
def generate_review(phone_query: str, audience: str = "a general buyer") -> dict:
    """Retrieve specifications, then write a review from them.

    The specification lookup always runs through the native
    `SpecificationAgent`: it is deterministic database access, and both
    backends need the same verified facts before any prose is written.
    """
    spec_agent = SpecificationAgent()
    spec_task = Task(
        name="specifications",
        description=f"Retrieve full specifications for '{phone_query}'",
        agent=spec_agent,
        inputs={"phone": phone_query},
    )
    spec_result = spec_agent.run(spec_task)

    if not spec_result.data.get("found"):
        return {
            "success": False,
            "phone": None,
            "review": spec_result.output,
            "framework": active_backend(),
            "agents": _serialise({"specifications": spec_result}),
        }

    phone_name = spec_result.data["phone_name"]
    backend = active_backend()

    if backend == "crewai":
        started = time.time()
        try:
            review_text = crewai_backend.run_review_crew(
                phone_name, spec_result.output, audience
            )
            elapsed = time.time() - started
            if review_text:
                return {
                    "success": True,
                    "phone": phone_name,
                    "specifications": spec_result.data["summary"],
                    "key_numbers": spec_result.data["key_numbers"],
                    "review": review_text,
                    "generated_by": "llm",
                    "framework": "crewai",
                    "agents": {
                        "specifications": spec_result.to_dict(),
                        **_crew_agent_trace(
                            ["Samsung Phone Data Specialist", "Senior Smartphone Reviewer"],
                            elapsed,
                        ),
                    },
                }
            logger.warning("CrewAI returned an empty review; using native backend")
        except Exception as exc:
            logger.warning("CrewAI review failed (%s); using native backend", exc)

    # Native path, also the fallback when CrewAI errors mid-run.
    review_agent = ReviewAgent()
    crew = Crew(
        name="ReviewCrew",
        tasks=[
            Task(
                name="review",
                description="Write a detailed product review from those specifications",
                agent=review_agent,
                inputs={"audience": audience},
                context_from=["specifications"],
            )
        ],
    )
    results = {"specifications": spec_result}
    results["review"] = review_agent.run(crew.tasks[0], results)

    return {
        "success": True,
        "phone": phone_name,
        "specifications": spec_result.data["summary"],
        "key_numbers": spec_result.data["key_numbers"],
        "review": results["review"].output,
        "generated_by": "llm" if results["review"].used_llm else "template",
        "framework": "native",
        "agents": _serialise(results),
    }


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def compare_phones(
    phone_a: str, phone_b: str, focus: str = "overall", query: str | None = None
) -> dict:
    """Fetch both spec sheets, then produce a head-to-head verdict."""
    spec_agent = SpecificationAgent()
    comparison_agent = ComparisonAgent()

    results: dict[str, AgentResult] = {}
    for name, model in (("specs_a", phone_a), ("specs_b", phone_b)):
        results[name] = spec_agent.run(
            Task(
                name=name,
                description=f"Retrieve specifications for '{model}'",
                agent=spec_agent,
                inputs={"phone": model},
            )
        )

    comparison_task = Task(
        name="comparison",
        description=f"Compare both phones on {focus}",
        agent=comparison_agent,
        inputs={
            "phones": [phone_a, phone_b],
            "focus": focus,
            "query": query or f"{phone_a} vs {phone_b}",
        },
        context_from=["specs_a", "specs_b"],
    )

    # The native agent builds the difference table and resolves both models,
    # which the CrewAI path then reuses as its grounding data.
    native_result = comparison_agent.run(comparison_task, results)

    if not native_result.data.get("compared"):
        results["comparison"] = native_result
        return {
            "success": False,
            "phones": native_result.data.get("phones", []),
            "focus": focus,
            "table": None,
            "comparison": native_result.output,
            "framework": active_backend(),
            "agents": _serialise(results),
        }

    names = native_result.data["phones"]
    table = native_result.data["table"]
    backend = active_backend()

    if backend == "crewai":
        started = time.time()
        try:
            # Compact summaries, not the full 58-row dumps. Two complete spec
            # sheets plus the table pushed the prompt past 4k tokens and the
            # two-task crew took over six minutes; the table already carries
            # every figure the verdict needs to cite.
            sheets = "\n\n".join(
                results[key].data["summary"] for key in ("specs_a", "specs_b")
                if results[key].data.get("found")
            )
            verdict = crewai_backend.run_comparison_crew(names, sheets, table, focus)
            elapsed = time.time() - started
            if verdict:
                return {
                    "success": True,
                    "phones": names,
                    "focus": focus,
                    "table": table,
                    "comparison": verdict,
                    "generated_by": "llm",
                    "framework": "crewai",
                    "agents": {
                        **_serialise(results),
                        **_crew_agent_trace(
                            [
                                "Samsung Phone Data Specialist",
                                "Smartphone Comparison Analyst",
                            ],
                            elapsed,
                        ),
                    },
                }
            logger.warning("CrewAI returned an empty comparison; using native output")
        except Exception as exc:
            logger.warning("CrewAI comparison failed (%s); using native output", exc)

    results["comparison"] = native_result
    return {
        "success": True,
        "phones": names,
        "focus": focus,
        "table": table,
        "comparison": native_result.output,
        "generated_by": "llm" if native_result.used_llm else "template",
        "framework": "native",
        "agents": _serialise(results),
    }


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def handle_request(query: str) -> dict:
    """Route a free-text request to the right crew.

    Two named models means a comparison; one means a review. This is what lets
    the API expose a single agent endpoint that accepts plain English.
    """
    with session_scope() as session:
        mentioned = extract_mentioned_phones(session, query)
        names = [p.name for p in mentioned]

    lowered = query.lower()
    focus = "overall"
    for candidate in (
        "camera", "battery", "display", "performance", "gaming",
        "price", "design", "storage",
    ):
        if candidate in lowered:
            focus = candidate
            break

    if len(names) >= 2:
        result = compare_phones(names[0], names[1], focus=focus, query=query)
        result["workflow"] = "comparison"
        return result

    if names:
        result = generate_review(names[0])
        result["workflow"] = "review"
        return result

    return {
        "success": False,
        "workflow": "none",
        "error": (
            "No known Samsung model was named. Mention a model such as "
            "'Galaxy S23 Ultra' to get a review, or two models to compare them."
        ),
    }
