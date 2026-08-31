"""Retrieval-augmented chatbot for Samsung phone questions.

Answering runs in four steps:

1. **Resolve** which handsets the question names (database lookup).
2. **Classify** the intent — spec lookup, comparison, superlative, price, … .
3. **Assemble context** the way that intent needs it: SQL ranking for
   superlatives, both spec sheets for comparisons, filtered vector search for
   everything else.
4. **Generate** an answer grounded in that context, with a deterministic
   template fallback when no LLM is loaded.

The context is always built from database rows, so the model is summarising
retrieved facts rather than recalling them, and every answer reports the sources
it used.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

import config
from src.database.db import session_scope
from src.database.models import Phone
from src.database.repository import (
    cheapest_phones,
    extract_mentioned_phones,
    get_all_phones,
    top_by_column,
)
from src.llm.provider import get_llm

from .documents import build_corpus
from .query_analysis import Intent, QueryAnalysis, analyze
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a knowledgeable Samsung smartphone specialist. Answer using ONLY "
    "the reference data provided; it comes from a GSMArena specification "
    "database. Never invent numbers. If the data does not cover something, say "
    "so plainly. Be specific, quote the actual figures, and keep the answer "
    "focused on what was asked. Write in clear prose, not bullet-point spam."
)

# Below this cosine similarity the corpus simply does not discuss the topic.
RELEVANCE_FLOOR = 0.25

# Query aspect (from query_analysis) -> the document aspects that answer it.
_ASPECT_TO_DOCUMENTS: dict[str, tuple[str, ...]] = {
    "camera": ("Camera",),
    "battery": ("Battery and charging",),
    "charging": ("Battery and charging",),
    "display": ("Display",),
    "performance": ("Performance",),
    "storage": ("Performance",),
    "software": ("Performance",),
    "design": ("Design and build",),
    "connectivity": ("Connectivity and features",),
    "price": ("Pricing", "Overview"),
}


def _document_aspects(aspects: list[str] | None) -> set[str]:
    """Document aspects worth prioritising for these query aspects."""
    if not aspects:
        return set()
    wanted: set[str] = set()
    # Only the top two query aspects count; beyond that the keyword scores are
    # weak enough that boosting on them would pull in unrelated passages.
    for aspect in aspects[:2]:
        wanted.update(_ASPECT_TO_DOCUMENTS.get(aspect, ()))
    return wanted


@dataclass
class ChatResponse:
    """An answer plus everything needed to audit how it was produced."""

    answer: str
    intent: str
    phones: list[str] = field(default_factory=list)
    aspects: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    generated_by: str = "template"

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "intent": self.intent,
            "phones": self.phones,
            "aspects": self.aspects,
            "sources": self.sources,
            "generated_by": self.generated_by,
        }


class SamsungChatbot:
    """RAG chatbot over the scraped Samsung phone database."""

    def __init__(self, vector_store: VectorStore | None = None, use_llm: bool | None = None):
        self.vector_store = vector_store or VectorStore()
        self.use_llm = config.USE_LLM if use_llm is None else use_llm
        self._ready = False

    # ------------------------------------------------------------------
    @staticmethod
    def _database_fingerprint(session: Session) -> str:
        """Identify the exact phone rows an index was built from."""
        rows = "|".join(
            f"{phone.id}:{phone.name}" for phone in get_all_phones(session)
        )
        return hashlib.sha256(rows.encode("utf-8")).hexdigest()[:16]

    def prepare(self, rebuild: bool = False) -> None:
        """Load or build the vector index. Safe to call more than once."""
        if self._ready and not rebuild:
            return

        with session_scope() as session:
            fingerprint = self._database_fingerprint(session)

        if not rebuild and self.vector_store.load(fingerprint=fingerprint):
            self._ready = True
            return

        with session_scope() as session:
            corpus = build_corpus(session)

        if not corpus:
            raise RuntimeError(
                "No phones in the database — run `python scripts/scrape.py` first"
            )

        self.vector_store.build(corpus)
        self.vector_store.save(fingerprint=fingerprint)
        self._ready = True

    # ------------------------------------------------------------------
    # Context builders, one per intent
    # ------------------------------------------------------------------
    def _retrieved_context(
        self,
        query: str,
        phones: list[Phone],
        top_k: int,
        aspects: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        phone_ids = [p.id for p in phones] if phones else None

        # Over-fetch, then narrow to the passages that actually answer the
        # question. Padding the context out to top_k with whatever ranked next
        # is actively harmful: asked for the S23's camera, the model was given
        # the display passage too and reported the screen's 120 Hz refresh rate
        # as a selfie-camera spec. Off-topic passages are dropped rather than
        # merely demoted.
        wanted = _document_aspects(aspects)
        fetch = top_k * 2 if wanted else top_k
        hits = self.vector_store.search(query, top_k=fetch, phone_ids=phone_ids)

        if wanted:
            on_topic = [hit for hit in hits if hit[0].aspect in wanted]
            # Only narrow when the corpus actually holds matching passages;
            # otherwise an unusual question would get an empty context.
            hits = on_topic or hits
        hits = hits[:top_k]

        blocks: list[str] = []
        sources: list[dict] = []
        for document, score in hits:
            if score < RELEVANCE_FLOOR and not phones:
                continue
            blocks.append(document.text)
            sources.append(
                {
                    "phone": document.phone_name,
                    "aspect": document.aspect,
                    "score": round(score, 4),
                }
            )
        return "\n\n".join(blocks), sources

    @staticmethod
    def _ranking_context(
        session: Session, analysis: QueryAnalysis
    ) -> tuple[str, list[dict], list[Phone]]:
        """Order the catalogue by the metric the superlative asks about."""
        aspect = analysis.primary_aspect or "battery"

        if aspect == "price":
            ranked = cheapest_phones(session, limit=6)
            header = "Samsung phones ranked from cheapest to most expensive (EUR):"
            describe = lambda p: (  # noqa: E731
                f"{p.name}: "
                + (
                    ", ".join(f"{pr.currency} {pr.amount:,.2f}" for pr in p.prices)
                    or "no listing"
                )
            )
        elif aspect == "performance" or analysis.ranking_column is None:
            # No single numeric column captures "performance", so hand the model
            # the chipset of each phone newest-first and let it reason.
            ranked = top_by_column(session, "release_year", limit=8)
            header = (
                "Samsung phones by generation (newest first) with their "
                "processor, RAM and GPU:"
            )
            describe = lambda p: (  # noqa: E731
                f"{p.name} ({p.release_year}): {p.chipset}; GPU {p.gpu}; "
                f"up to {p.max_ram_gb}GB RAM"
            )
        else:
            column = analysis.ranking_column
            ranked = top_by_column(
                session, column, limit=6, descending=analysis.higher_is_better
            )
            label = {
                "battery_capacity_mah": ("battery capacity", "{} mAh"),
                "main_camera_mp": ("main camera resolution", "{} MP"),
                "display_size_inches": ("display size", "{} inches"),
                "max_storage_gb": ("maximum storage", "{} GB"),
                "charging_watts": ("charging speed", "{} W"),
                "weight_g": ("weight, lightest first", "{} g"),
            }.get(column, (column, "{}"))
            direction = "highest" if analysis.higher_is_better else "lowest"
            header = f"Samsung phones ranked by {label[0]} ({direction} first):"
            describe = lambda p: (  # noqa: E731
                f"{p.name}: {label[1].format(getattr(p, column))}"
                + (f" (battery {p.battery_capacity_mah} mAh)"
                   if column != "battery_capacity_mah" and p.battery_capacity_mah
                   else "")
            )

        lines = [f"{i}. {describe(p)}" for i, p in enumerate(ranked, start=1)]
        context = header + "\n" + "\n".join(lines)
        sources = [{"phone": p.name, "aspect": f"ranking:{aspect}"} for p in ranked]
        return context, sources, ranked

    def _comparison_context(
        self, query: str, phones: list[Phone], analysis: QueryAnalysis
    ) -> tuple[str, list[dict]]:
        """Put both spec sheets side by side, focused on the asked-about aspect."""
        aspect = analysis.primary_aspect
        blocks = ["Specification data for the phones being compared:"]
        sources: list[dict] = []

        for phone in phones:
            blocks.append(phone.spec_summary())
            sources.append({"phone": phone.name, "aspect": "spec_sheet"})

        if aspect:
            # Retrieval adds the narrative detail the flat summary omits.
            retrieved, retrieved_sources = self._retrieved_context(
                f"{aspect} {query}",
                phones,
                top_k=len(phones) * 2,
                aspects=analysis.aspects,
            )
            if retrieved:
                blocks.append(f"Additional {aspect} detail:\n{retrieved}")
                sources.extend(retrieved_sources)

        return "\n\n".join(blocks), sources

    @staticmethod
    def _catalogue_context(session: Session) -> tuple[str, list[dict]]:
        phones = get_all_phones(session)
        lines = [
            f"- {p.name} ({p.release_year}): {p.display_size_inches}\" display, "
            f"{p.chipset}, {p.battery_capacity_mah} mAh"
            for p in phones
        ]
        context = f"The database contains {len(phones)} Samsung phones:\n" + "\n".join(lines)
        return context, [{"phone": p.name, "aspect": "catalogue"} for p in phones]

    # ------------------------------------------------------------------
    def chat(self, query: str, top_k: int | None = None) -> ChatResponse:
        """Answer one question."""
        self.prepare()
        top_k = top_k or config.RAG_TOP_K
        query = (query or "").strip()

        if not query:
            return ChatResponse(
                answer="Please ask a question about Samsung phones.",
                intent=Intent.GENERAL.value,
            )

        with session_scope() as session:
            mentioned = extract_mentioned_phones(session, query)
            analysis = analyze(query, mentioned_phone_count=len(mentioned))

            if analysis.intent == Intent.SUPERLATIVE:
                context, sources, ranked = self._ranking_context(session, analysis)
                focus_phones = ranked[:3]
            elif analysis.intent == Intent.COMPARISON and mentioned:
                context, sources = self._comparison_context(query, mentioned, analysis)
                focus_phones = mentioned
            elif analysis.intent == Intent.LIST:
                context, sources = self._catalogue_context(session)
                focus_phones = []
            elif analysis.intent == Intent.RECOMMENDATION:
                ranking_context, ranking_sources, ranked = self._ranking_context(
                    session, analysis
                )
                retrieved, retrieved_sources = self._retrieved_context(
                    query, [], top_k=top_k, aspects=analysis.aspects
                )
                context = f"{ranking_context}\n\nRelevant details:\n{retrieved}"
                sources = ranking_sources + retrieved_sources
                focus_phones = ranked[:3]
            else:
                context, sources = self._retrieved_context(
                    query, mentioned, top_k, aspects=analysis.aspects
                )
                focus_phones = mentioned

            phone_names = [p.name for p in (focus_phones or mentioned)]

            if not context.strip():
                return ChatResponse(
                    answer=(
                        "I could not find anything about that in the phone "
                        "database. It covers 15 Samsung models — try asking "
                        "about their displays, cameras, batteries, processors "
                        "or prices."
                    ),
                    intent=analysis.intent.value,
                    aspects=analysis.aspects,
                )

            answer, generated_by = self._generate(query, context, analysis, focus_phones)

        return ChatResponse(
            answer=answer,
            intent=analysis.intent.value,
            phones=phone_names,
            aspects=analysis.aspects,
            sources=sources,
            generated_by=generated_by,
        )

    # ------------------------------------------------------------------
    def _generate(
        self,
        query: str,
        context: str,
        analysis: QueryAnalysis,
        phones: list[Phone],
    ) -> tuple[str, str]:
        """Produce prose from the assembled context."""
        llm = get_llm() if self.use_llm else None

        if llm is None:
            return self._template_answer(query, context, analysis, phones), "template"

        instruction = {
            Intent.COMPARISON: (
                "Compare the phones directly. State which is stronger on the "
                "asked-about dimension and why, citing the specific figures."
            ),
            Intent.SUPERLATIVE: (
                "Name the winner first, give its figure, then briefly mention "
                "the runners-up for context."
            ),
            Intent.PRICE: "Report the prices per currency.",
            Intent.RECOMMENDATION: (
                "Give one clear recommendation with the reasoning behind it."
            ),
        }.get(analysis.intent, "Answer the question precisely using the data.")

        user_prompt = (
            f"Reference data:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"{instruction}"
        )

        try:
            answer = llm.complete(SYSTEM_PROMPT, user_prompt)
            if answer.strip():
                return answer.strip(), f"llm:{llm.model_name}"
            logger.warning("LLM returned an empty answer; using template")
        except Exception as exc:
            logger.warning("Generation failed (%s); using template", exc)

        return self._template_answer(query, context, analysis, phones), "template"

    @staticmethod
    def _template_answer(
        query: str, context: str, analysis: QueryAnalysis, phones: list[Phone]
    ) -> str:
        """Deterministic answer assembled straight from retrieved facts.

        This is not a stub: with no LLM the system still answers correctly, it
        just reads as structured data rather than prose.
        """
        aspect = analysis.primary_aspect

        if analysis.intent == Intent.SUPERLATIVE:
            return (
                f"Based on the specification database:\n\n{context}\n\n"
                f"The first entry is the strongest on {aspect or 'this metric'}."
            )

        if analysis.intent == Intent.COMPARISON and len(phones) >= 2:
            names = " vs ".join(p.name for p in phones)
            return f"Comparing {names}:\n\n{context}"

        if phones:
            header = f"Here is what the database holds for {phones[0].name}"
            if aspect:
                header += f" regarding {aspect}"
            return f"{header}:\n\n{context}"

        return f"Relevant specifications:\n\n{context}"

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {"vector_store": self.vector_store.stats(), "use_llm": self.use_llm}


_chatbot: SamsungChatbot | None = None


def get_chatbot() -> SamsungChatbot:
    """Shared chatbot instance (one vector index per process)."""
    global _chatbot
    if _chatbot is None:
        _chatbot = SamsungChatbot()
        _chatbot.prepare()
    return _chatbot
