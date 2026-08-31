"""The specialist agents: data retrieval, review writing, comparison.

`SpecificationAgent` is the only one that touches the database. The writing
agents work purely from what it hands them, which is what keeps a generated
review tied to real scraped data instead of model recall.
"""

from __future__ import annotations

import logging

from src.database.db import session_scope
from src.database.models import Phone
from src.database.repository import (
    extract_mentioned_phones,
    find_phone,
    get_all_phones,
    top_by_column,
)

from .base import Agent, AgentResult, Task

logger = logging.getLogger(__name__)


def _format_specs(phone: Phone) -> str:
    """Full spec sheet as text, grouped the way GSMArena groups it."""
    sections: dict[str, list[str]] = {}
    for spec in phone.specifications:
        sections.setdefault(spec.category, []).append(f"  {spec.name}: {spec.value}")

    blocks = [f"=== {phone.name} ==="]
    for category, rows in sections.items():
        blocks.append(f"[{category}]")
        blocks.extend(rows)

    if phone.prices:
        listed = ", ".join(f"{p.currency} {p.amount:,.2f}" for p in phone.prices)
        blocks.append(f"[Market price]\n  {listed}")

    return "\n".join(blocks)


def _key_numbers(phone: Phone) -> dict:
    """The parsed numeric fields, for agents that need to reason over values."""
    return {
        "name": phone.name,
        "release_year": phone.release_year,
        "display_inches": phone.display_size_inches,
        "refresh_rate_hz": phone.display_refresh_rate_hz,
        "chipset": phone.chipset,
        "max_ram_gb": phone.max_ram_gb,
        "max_storage_gb": phone.max_storage_gb,
        "main_camera_mp": phone.main_camera_mp,
        "selfie_camera_mp": phone.selfie_camera_mp,
        "battery_mah": phone.battery_capacity_mah,
        "charging_w": phone.charging_watts,
        "weight_g": phone.weight_g,
        "prices": [{"currency": p.currency, "amount": p.amount} for p in phone.prices],
    }


class SpecificationAgent(Agent):
    """Retrieves detailed phone specifications from the database.

    Produces no prose of its own — its output is the factual payload every
    downstream agent is required to build on.
    """

    def __init__(self) -> None:
        super().__init__(
            name="SpecificationAgent",
            role="a technical data specialist for Samsung smartphones",
            goal=(
                "retrieve complete, accurate specifications from the phone "
                "database and present them without interpretation"
            ),
            backstory=(
                "You work directly with the scraped GSMArena database and are "
                "trusted because you never guess at a value."
            ),
        )

    def execute(self, task: Task, context: dict[str, AgentResult]) -> AgentResult:
        query = task.inputs.get("phone") or task.inputs.get("query", "")

        with session_scope() as session:
            phone = find_phone(session, query)
            if phone is None:
                available = ", ".join(p.name for p in get_all_phones(session))
                return AgentResult(
                    agent=self.name,
                    output=f"No phone matching '{query}'. Available models: {available}",
                    data={"found": False, "query": query},
                )

            # Peer context lets the review agent say "large for its generation"
            # instead of quoting a number with nothing to measure it against.
            battery_ranking = [
                (p.name, p.battery_capacity_mah)
                for p in top_by_column(session, "battery_capacity_mah", limit=15)
            ]
            position = next(
                (i for i, (n, _) in enumerate(battery_ranking, 1) if n == phone.name),
                None,
            )

            spec_sheet = _format_specs(phone)
            numbers = _key_numbers(phone)
            numbers["battery_rank_of"] = (position, len(battery_ranking))

            return AgentResult(
                agent=self.name,
                output=spec_sheet,
                data={
                    "found": True,
                    "phone_id": phone.id,
                    "phone_name": phone.name,
                    "summary": phone.spec_summary(),
                    "key_numbers": numbers,
                    "spec_count": len(phone.specifications),
                },
            )


class ReviewAgent(Agent):
    """Writes a full product review from the retrieved specifications."""

    SECTIONS = (
        "Design and build",
        "Display",
        "Performance",
        "Cameras",
        "Battery and charging",
        "Value for money",
        "Verdict",
    )

    def __init__(self) -> None:
        super().__init__(
            name="ReviewAgent",
            role="a senior smartphone reviewer",
            goal=(
                "turn a specification sheet into an informative, balanced review "
                "that explains what the numbers mean for everyday use"
            ),
            backstory=(
                "You have reviewed phones for a decade. You are even-handed: "
                "you name real weaknesses as readily as strengths, and every "
                "claim you make traces back to a specification."
            ),
        )

    def execute(self, task: Task, context: dict[str, AgentResult]) -> AgentResult:
        spec_result = next(
            (r for r in context.values() if r.data.get("found")), None
        )
        if spec_result is None:
            return AgentResult(
                agent=self.name,
                output="No specification data was provided, so no review can be written.",
                data={"written": False},
            )

        phone_name = spec_result.data["phone_name"]
        numbers = spec_result.data["key_numbers"]
        audience = task.inputs.get("audience", "a general buyer")

        prompt = (
            f"Write a detailed product review of the {phone_name} for {audience}.\n\n"
            f"Full specifications:\n{spec_result.output}\n\n"
            f"Parsed key figures: {numbers}\n\n"
            "Cover these sections in order, one short paragraph each: "
            + ", ".join(self.SECTIONS)
            + ". Quote concrete specifications to support each judgement. "
            "Mention at least one genuine limitation. Do not invent benchmark "
            "scores, battery-life hours, or anything not present above."
        )

        output, used_llm = self.think(prompt, max_new_tokens=900)

        if not used_llm:
            output = self._structured_review(phone_name, spec_result)

        return AgentResult(
            agent=self.name,
            output=output,
            data={
                "written": True,
                "phone_name": phone_name,
                "sections": list(self.SECTIONS),
            },
            used_llm=used_llm,
        )

    @staticmethod
    def _structured_review(phone_name: str, spec_result: AgentResult) -> str:
        """Fact-derived review used when no LLM is loaded."""
        n = spec_result.data["key_numbers"]
        rank, total = n.get("battery_rank_of", (None, None))

        battery_line = (
            f"The {n['battery_mah']} mAh cell ranks {rank} of {total} phones in "
            f"this database"
            if n.get("battery_mah") and rank
            else "Battery capacity is not recorded"
        )
        charging = f", charging at up to {n['charging_w']}W." if n.get("charging_w") else "."
        price = (
            ", ".join(f"{p['currency']} {p['amount']:,.2f}" for p in n.get("prices", []))
            or "not listed"
        )

        return "\n\n".join(
            [
                f"# {phone_name} — Review",
                f"**Design and build.** The phone weighs {n.get('weight_g')} g.",
                f"**Display.** A {n.get('display_inches')}-inch panel running at up to "
                f"{n.get('refresh_rate_hz')} Hz.",
                f"**Performance.** Built on the {n.get('chipset')}, with up to "
                f"{n.get('max_ram_gb')} GB of RAM and {n.get('max_storage_gb')} GB of storage.",
                f"**Cameras.** A {n.get('main_camera_mp')} MP main sensor and a "
                f"{n.get('selfie_camera_mp')} MP front camera.",
                f"**Battery and charging.** {battery_line}{charging}",
                f"**Value for money.** Current listings: {price}.",
                f"**Verdict.** Released in {n.get('release_year')}, the {phone_name} "
                "is characterised by the specifications above.",
            ]
        )


class ComparisonAgent(Agent):
    """Produces a head-to-head comparison of two or more handsets."""

    def __init__(self) -> None:
        super().__init__(
            name="ComparisonAgent",
            role="a comparison analyst for smartphones",
            goal=(
                "explain concretely how two phones differ and which suits which "
                "kind of buyer"
            ),
            backstory=(
                "You are known for turning spec tables into a clear "
                "recommendation without hedging."
            ),
        )

    def execute(self, task: Task, context: dict[str, AgentResult]) -> AgentResult:
        models = task.inputs.get("phones") or []
        focus = task.inputs.get("focus", "overall")

        with session_scope() as session:
            phones: list[Phone] = []
            for model in models:
                phone = find_phone(session, model)
                if phone is not None:
                    phones.append(phone)

            if len(phones) < 2:
                query = task.inputs.get("query", " ".join(models))
                phones = extract_mentioned_phones(session, query)

            if len(phones) < 2:
                return AgentResult(
                    agent=self.name,
                    output="At least two recognisable phone models are needed to compare.",
                    data={"compared": False, "requested": models},
                )

            table = self._difference_table(phones)
            sheets = "\n\n".join(_format_specs(p) for p in phones)
            names = " vs ".join(p.name for p in phones)

            prompt = (
                f"Compare these phones with a focus on {focus}.\n\n"
                f"Side-by-side key figures:\n{table}\n\n"
                f"Full specifications:\n{sheets}\n\n"
                f"Explain the meaningful differences in {focus}, state which "
                "phone wins and why, and note who should pick the other one. "
                "Cite the actual figures."
            )

            output, used_llm = self.think(prompt, max_new_tokens=700)
            if not used_llm:
                output = f"# {names} — {focus}\n\n{table}"

            return AgentResult(
                agent=self.name,
                output=output,
                data={
                    "compared": True,
                    "phones": [p.name for p in phones],
                    "focus": focus,
                    "table": table,
                },
                used_llm=used_llm,
            )

    @staticmethod
    def _difference_table(phones: list[Phone]) -> str:
        """Aligned text table of the fields that usually differ."""
        rows = [
            ("Released", lambda p: str(p.release_year or "—")),
            ("Display", lambda p: f'{p.display_size_inches or "—"}"'),
            ("Refresh rate", lambda p: f'{p.display_refresh_rate_hz or "—"} Hz'),
            ("Chipset", lambda p: p.chipset or "—"),
            ("GPU", lambda p: p.gpu or "—"),
            ("Max RAM", lambda p: f'{p.max_ram_gb or "—"} GB'),
            ("Max storage", lambda p: f'{p.max_storage_gb or "—"} GB'),
            ("Main camera", lambda p: f'{p.main_camera_mp or "—"} MP'),
            ("Selfie camera", lambda p: f'{p.selfie_camera_mp or "—"} MP'),
            ("Battery", lambda p: f'{p.battery_capacity_mah or "—"} mAh'),
            ("Charging", lambda p: f'{p.charging_watts or "—"} W'),
            ("Weight", lambda p: f'{p.weight_g or "—"} g'),
        ]

        label_width = max(len(label) for label, _ in rows) + 2

        # Long values (chipset strings run past 40 characters) would otherwise
        # spill into the next column, so each column is sized to its widest
        # cell, not just to the phone's name.
        widths = [
            max(len(phone.name), *(len(getter(phone)) for _, getter in rows)) + 3
            for phone in phones
        ]

        header = "Spec".ljust(label_width) + "".join(
            phone.name.ljust(width) for phone, width in zip(phones, widths)
        )
        lines = [header, "-" * len(header.rstrip())]

        for label, getter in rows:
            line = label.ljust(label_width) + "".join(
                getter(phone).ljust(width) for phone, width in zip(phones, widths)
            )
            lines.append(line.rstrip())

        return "\n".join(lines)
