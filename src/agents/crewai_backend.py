"""CrewAI implementation of the multi-agent system.

The brief names CrewAI, AutoGen or LangChain, and this module is the CrewAI
path: real `Agent`, `Task` and `Crew` objects, `Process.sequential`, a database
`BaseTool`, and task-to-task context passing.

The one piece of custom code is `LocalCrewLLM`. CrewAI routes model calls
through LiteLLM, which expects a hosted provider; the brief also asks for
open-source models. Subclassing CrewAI's `BaseLLM` lets the crew drive the same
local Hugging Face model the rest of the system uses, with no API key and no
separate inference server.

Import is guarded throughout: if CrewAI is not installed, `is_available()`
returns False and `src/agents/crew.py` falls back to the native orchestrator.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Type

# Set before CrewAI is imported. On a first run CrewAI otherwise prompts on
# stdin — "Would you like to view your execution traces? [y/N] (20s timeout)" —
# which would hang an API worker for 20 seconds with nobody there to answer.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import config
from src.database.db import session_scope
from src.database.repository import find_phone, get_all_phones
from src.llm.provider import get_llm

logger = logging.getLogger(__name__)

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    try:  # import path differs across CrewAI versions
        from crewai import BaseLLM
    except ImportError:
        from crewai.llms.base_llm import BaseLLM

    CREWAI_AVAILABLE = True
    CREWAI_IMPORT_ERROR: str | None = None
except Exception as exc:  # ImportError, or a version mismatch on a sub-import
    CREWAI_AVAILABLE = False
    CREWAI_IMPORT_ERROR = str(exc)
    logger.info("CrewAI unavailable (%s); native orchestrator will be used", exc)


def is_available() -> bool:
    """True when CrewAI is importable and an LLM is loadable.

    CrewAI agents have no template fallback — an agent with no working model
    cannot produce anything — so a missing LLM disqualifies this backend just
    as a missing CrewAI install does.
    """
    return CREWAI_AVAILABLE and get_llm() is not None


def status() -> dict:
    return {
        "crewai_installed": CREWAI_AVAILABLE,
        "import_error": CREWAI_IMPORT_ERROR,
        "llm_loaded": get_llm() is not None if CREWAI_AVAILABLE else False,
        "available": is_available(),
    }


# --------------------------------------------------------------------------
# Everything below needs CrewAI's base classes at definition time.
# --------------------------------------------------------------------------
if CREWAI_AVAILABLE:

    class LocalCrewLLM(BaseLLM):
        """Routes CrewAI's model calls to the shared local Qwen model."""

        # Crew tasks produce whole reviews, not single answers, so they need a
        # larger budget than the chatbot's default or the last section gets
        # truncated mid-sentence. ClassVar keeps Pydantic from treating this as
        # a model field.
        CREW_MAX_NEW_TOKENS: ClassVar[int] = 1024

        def __init__(self, temperature: float | None = None):
            # BaseLLM is a Pydantic model in CrewAI 1.x. `provider`/`is_litellm`
            # keep the crew from routing this model through LiteLLM, which would
            # try to reach a hosted endpoint.
            super().__init__(
                model=config.LLM_MODEL,
                temperature=(
                    config.LLM_TEMPERATURE if temperature is None else temperature
                ),
                provider="local",
                is_litellm=False,
            )

        def call(
            self,
            messages: str | list[dict[str, str]],
            tools: list[dict] | None = None,
            callbacks: list[Any] | None = None,
            available_functions: dict[str, Any] | None = None,
            from_task: Any | None = None,
            from_agent: Any | None = None,
            response_model: Any | None = None,
        ) -> str:
            llm = get_llm()
            if llm is None:
                raise RuntimeError("Local LLM is not available")

            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

            # CrewAI may pass roles the chat template does not know; anything
            # that is not system/assistant is safe to treat as user turn text.
            normalised = [
                {
                    "role": m.get("role") if m.get("role") in ("system", "assistant") else "user",
                    "content": str(m.get("content", "")),
                }
                for m in messages
                if m.get("content")
            ]
            if not normalised:
                return ""

            return llm.chat(normalised, max_new_tokens=self.CREW_MAX_NEW_TOKENS)

        def supports_function_calling(self) -> bool:
            # A 1.5B model cannot emit reliable JSON tool calls. Reporting False
            # keeps CrewAI on its text protocol instead of silently dropping
            # malformed calls.
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            # Matches the truncation limit in LocalLLM.chat().
            return 8192

    class PhoneLookupInput(BaseModel):
        phone: str = Field(
            ..., description="Samsung model name, for example 'Galaxy S23 Ultra'"
        )

    class PhoneSpecificationTool(BaseTool):
        """CrewAI tool exposing the scraped database to the agents."""

        name: str = "phone_specification_lookup"
        description: str = (
            "Look up the complete specification sheet for a Samsung phone from "
            "the GSMArena database. Input is the model name, e.g. 'Galaxy S23 "
            "Ultra'. Returns display, chipset, camera, battery and price data."
        )
        args_schema: Type[BaseModel] = PhoneLookupInput

        def _run(self, phone: str) -> str:
            # Imported here to avoid a circular import at module load.
            from .specialists import _format_specs

            with session_scope() as session:
                match = find_phone(session, phone)
                if match is None:
                    available = ", ".join(p.name for p in get_all_phones(session))
                    return f"No phone matching '{phone}'. Available models: {available}"
                return _format_specs(match)

    # ------------------------------------------------------------------
    # Agent definitions
    # ------------------------------------------------------------------
    def _shared_llm() -> "LocalCrewLLM":
        return LocalCrewLLM()

    def build_specification_agent(llm: "LocalCrewLLM") -> "Agent":
        return Agent(
            role="Samsung Phone Data Specialist",
            goal=(
                "Retrieve complete and accurate specifications for a requested "
                "Samsung phone from the GSMArena database, without interpreting "
                "or embellishing them."
            ),
            backstory=(
                "You maintain the scraped specification database and are trusted "
                "precisely because you never guess at a value. You report what "
                "the database says and nothing more."
            ),
            tools=[PhoneSpecificationTool()],
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )

    def build_review_agent(llm: "LocalCrewLLM") -> "Agent":
        return Agent(
            role="Senior Smartphone Reviewer",
            goal=(
                "Turn a specification sheet into an informative, balanced review "
                "that explains what the numbers mean in everyday use."
            ),
            backstory=(
                "You have reviewed phones for a decade. You are even-handed: you "
                "name real weaknesses as readily as strengths, and every claim "
                "you make traces back to a specification you were given."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=2,
        )

    def build_comparison_agent(llm: "LocalCrewLLM") -> "Agent":
        return Agent(
            role="Smartphone Comparison Analyst",
            goal=(
                "Explain concretely how two phones differ and which buyer each "
                "one suits."
            ),
            backstory=(
                "You are known for turning spec tables into a clear "
                "recommendation without hedging."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=2,
        )

    # ------------------------------------------------------------------
    # Crews
    # ------------------------------------------------------------------
    def run_review_crew(phone_name: str, spec_sheet: str, audience: str) -> str:
        """SpecificationAgent -> ReviewAgent, wired as a sequential CrewAI crew.

        `spec_sheet` is passed in already retrieved. The tool stays attached to
        the specification agent so a capable model calls it itself, but the
        authoritative data is also placed in the task description: at 1.5B
        parameters the local model cannot be trusted to drive CrewAI's text tool
        protocol, and a review built on a failed lookup would be fiction. This
        keeps the crew's output grounded regardless of model size.
        """
        llm = _shared_llm()
        spec_agent = build_specification_agent(llm)
        review_agent = build_review_agent(llm)

        gather = Task(
            description=(
                f"Report the key specifications of the {phone_name}.\n\n"
                f"Database record:\n{spec_sheet}\n\n"
                # Held to one line per category on purpose: this output is
                # prepended to the review task's prompt as context, and on a
                # 6 GB GPU generation slows sharply as the prompt grows. A
                # verbose summary here cost more time than the review itself.
                "Give exactly one short line each for display, chipset, memory, "
                "cameras, battery and price. Under 120 words total. Copy the "
                "figures verbatim and add nothing that is not present."
            ),
            expected_output=(
                "Six short lines: display, chipset, memory, cameras, battery, price."
            ),
            agent=spec_agent,
        )

        write = Task(
            description=(
                f"Write a detailed product review of the {phone_name} for "
                f"{audience}.\n\n"
                # The authoritative record is repeated here rather than relying
                # on the previous task's restatement of it. A 1.5B model
                # paraphrasing 58 specs introduces errors, and a review built
                # only on that paraphrase compounds them — an early version
                # described a 234 g phone as "just under 200 grams".
                f"Authoritative database record (use these exact figures):\n"
                f"{spec_sheet}\n\n"
                "The previous task's summary is available as context, but where "
                "it disagrees with the record above, the record wins.\n\n"
                "Cover, in order, one short paragraph each: Design and build, "
                "Display, Performance, Cameras, Battery and charging, Value for "
                "money, Verdict. Quote concrete figures. Mention at least one "
                "genuine limitation. Invent nothing."
            ),
            expected_output="A seven-paragraph product review citing real specifications.",
            agent=review_agent,
            context=[gather],  # CrewAI feeds task 1's output into task 2
        )

        crew = Crew(
            agents=[spec_agent, review_agent],
            tasks=[gather, write],
            process=Process.sequential,
            verbose=False,
            tracing=False,
        )

        return str(crew.kickoff()).strip()

    def run_comparison_crew(
        names: list[str], spec_sheets: str, table: str, focus: str
    ) -> str:
        """Two retrieval tasks feeding a comparison task."""
        llm = _shared_llm()
        spec_agent = build_specification_agent(llm)
        comparison_agent = build_comparison_agent(llm)

        gather = Task(
            description=(
                f"Report the specifications relevant to {focus} for "
                f"{' and '.join(names)}.\n\n"
                f"Database records:\n{spec_sheets}\n\n"
                f"Side-by-side key figures:\n{table}\n\n"
                "List only the figures that actually differ between the two "
                "phones, and note explicitly which ones are identical. Keep it "
                "under 120 words — this feeds the next task's prompt."
            ),
            expected_output=f"A short factual list of {focus} differences.",
            agent=spec_agent,
        )

        analyse = Task(
            description=(
                f"Compare {' and '.join(names)} on {focus}.\n\n"
                # Same reasoning as the review crew: the table is the ground
                # truth, because a paraphrase of it from the previous task
                # invented chipsets and battery capacities that do not exist.
                f"Authoritative side-by-side figures (use these exact values):\n"
                f"{table}\n\n"
                "The previous task's notes are available as context, but where "
                "they disagree with the table above, the table wins.\n\n"
                "State which phone wins and why, citing the actual numbers, and "
                "say who should pick the other one instead. Where a figure is "
                "identical for both phones, say so rather than inventing a "
                "difference."
            ),
            expected_output=f"A clear {focus} comparison with a verdict.",
            agent=comparison_agent,
            context=[gather],
        )

        crew = Crew(
            agents=[spec_agent, comparison_agent],
            tasks=[gather, analyse],
            process=Process.sequential,
            verbose=False,
            tracing=False,
        )

        return str(crew.kickoff()).strip()
