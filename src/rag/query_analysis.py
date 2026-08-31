"""Classify a user question before retrieval runs.

Pure vector search is weak on two of the three sample queries. Embeddings have
no notion of *ranking*, so "which phone has the best battery life?" retrieves
passages that merely talk about batteries rather than the one with the largest
number; and a comparison question mentions two models, so a single nearest-
neighbour lookup returns whichever one the wording happens to favour.

So the question is routed first: superlatives become SQL `ORDER BY`, comparisons
fetch both phones explicitly, and only open-ended questions rely on similarity
alone. The retrieved text still grounds the final answer in every case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    SPEC_LOOKUP = "spec_lookup"       # "camera specs of the S23"
    COMPARISON = "comparison"          # "S23 vs S22 performance"
    SUPERLATIVE = "superlative"        # "best battery life"
    PRICE = "price"                    # "how much is the S24"
    RECOMMENDATION = "recommendation"  # "which should I buy for photography"
    LIST = "list"                      # "what phones do you know about"
    GENERAL = "general"


# aspect -> (keywords, ranking column, higher_is_better)
ASPECTS: dict[str, tuple[tuple[str, ...], str | None, bool]] = {
    "battery": (
        ("battery", "batteries", "mah", "charge", "charging", "endurance",
         "battery life", "lasts", "power"),
        "battery_capacity_mah",
        True,
    ),
    "camera": (
        ("camera", "cameras", "photo", "photography", "megapixel", "mp",
         "selfie", "zoom", "lens", "video", "recording", "telephoto",
         "ultrawide", "picture"),
        "main_camera_mp",
        True,
    ),
    "display": (
        ("display", "screen", "resolution", "refresh", "amoled", "oled",
         "nits", "inch", "inches", "brightness", "panel", "ppi"),
        "display_size_inches",
        True,
    ),
    "performance": (
        ("performance", "processor", "chipset", "cpu", "gpu", "snapdragon",
         "exynos", "speed", "fast", "faster", "fastest", "gaming", "ram",
         "benchmark", "powerful", "chip"),
        None,  # ranked by generation, handled specially
        True,
    ),
    "storage": (
        ("storage", "internal", "gb", "tb", "memory", "card slot", "expandable"),
        "max_storage_gb",
        True,
    ),
    "price": (
        ("price", "prices", "cost", "costs", "cheap", "cheapest", "expensive",
         "budget", "affordable", "worth", "value"),
        None,  # prices live in their own table
        False,
    ),
    "design": (
        ("design", "build", "weight", "weighs", "light", "lightest", "heavy",
         "dimensions", "thin", "color", "colours", "colors", "material",
         "waterproof", "ip68", "durable"),
        "weight_g",
        False,  # lighter is better
    ),
    "charging": (
        ("charging speed", "fast charging", "watt", "watts", "w charging",
         "wireless charging"),
        "charging_watts",
        True,
    ),
    "connectivity": (
        ("wifi", "wi-fi", "bluetooth", "nfc", "usb", "headphone", "jack",
         "speaker", "5g", "sensors", "fingerprint"),
        None,
        True,
    ),
    "software": (
        ("android", "one ui", "os", "update", "updates", "software"),
        None,
        True,
    ),
}

_SUPERLATIVE_WORDS = (
    "best", "worst", "top", "most", "highest", "lowest", "largest", "biggest",
    "smallest", "longest", "shortest", "cheapest", "priciest", "greatest",
    "maximum", "minimum", "fastest", "slowest", "lightest", "heaviest",
    "which phone", "which samsung", "which model", "rank",
)

_COMPARISON_WORDS = (
    " vs ", " vs. ", "versus", "compare", "comparison", "compared to",
    "difference between", "differences between", "better than", "or the",
)

_LIST_WORDS = (
    "what phones", "which phones do you", "list all", "list the phones",
    "how many phones", "what models", "available phones", "what do you know",
)

_RECOMMENDATION_WORDS = (
    "should i buy", "recommend", "suggestion", "suit me", "good for me",
    "which one should", "worth buying", "best for",
)


@dataclass
class QueryAnalysis:
    """Structured reading of a user question."""

    query: str
    intent: Intent
    aspects: list[str] = field(default_factory=list)
    ranking_column: str | None = None
    higher_is_better: bool = True

    @property
    def primary_aspect(self) -> str | None:
        return self.aspects[0] if self.aspects else None

    def __repr__(self) -> str:
        return f"<QueryAnalysis {self.intent.value} aspects={self.aspects}>"


def detect_aspects(query: str) -> list[str]:
    """Which spec areas the question touches, most relevant first."""
    lowered = f" {query.lower()} "
    hits: list[tuple[int, str]] = []

    for aspect, (keywords, _, _) in ASPECTS.items():
        score = 0
        for keyword in keywords:
            # Multi-word keywords are matched as substrings; single words need
            # boundaries so "mp" does not fire inside "important".
            if " " in keyword:
                if keyword in lowered:
                    score += 2
            elif re.search(rf"\b{re.escape(keyword)}\b", lowered):
                score += 1
        if score:
            hits.append((score, aspect))

    hits.sort(key=lambda item: item[0], reverse=True)
    return [aspect for _, aspect in hits]


def analyze(query: str, mentioned_phone_count: int = 0) -> QueryAnalysis:
    """Classify a question into an intent plus the aspects it asks about.

    `mentioned_phone_count` comes from the caller's database lookup: two named
    models is the strongest possible signal for a comparison, stronger than any
    keyword, since "S23 or S22 for gaming?" contains no comparison word at all.
    """
    lowered = f" {query.lower()} "
    aspects = detect_aspects(query)

    ranking_column = None
    higher_is_better = True
    if aspects:
        _, ranking_column, higher_is_better = ASPECTS[aspects[0]]

    is_superlative = any(word in lowered for word in _SUPERLATIVE_WORDS)
    is_comparison = any(word in lowered for word in _COMPARISON_WORDS)

    if mentioned_phone_count >= 2:
        intent = Intent.COMPARISON
    elif is_comparison and mentioned_phone_count >= 1:
        intent = Intent.COMPARISON
    elif any(word in lowered for word in _LIST_WORDS):
        intent = Intent.LIST
    elif is_superlative and mentioned_phone_count == 0:
        # A superlative naming one phone ("is the S23 the best?") is still a
        # question about that phone, not a ranking over the catalogue.
        intent = Intent.SUPERLATIVE
    elif any(word in lowered for word in _RECOMMENDATION_WORDS):
        intent = Intent.RECOMMENDATION
    elif "price" in aspects[:1] and mentioned_phone_count >= 1:
        intent = Intent.PRICE
    elif mentioned_phone_count >= 1:
        intent = Intent.SPEC_LOOKUP
    else:
        intent = Intent.GENERAL

    return QueryAnalysis(
        query=query,
        intent=intent,
        aspects=aspects,
        ranking_column=ranking_column,
        higher_is_better=higher_is_better,
    )
