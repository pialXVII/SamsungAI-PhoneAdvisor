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

import difflib
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
        # "charge"/"charging" belong to the charging aspect below, so a
        # question about charging speed ranks by watts, not by capacity.
        ("battery", "batteries", "mah", "endurance", "battery life", "lasts",
         "power"),
        "battery_capacity_mah",
        True,
    ),
    "camera": (
        ("camera", "cameras", "photo", "photography", "megapixel", "mp",
         "selfie", "zoom", "lens", "video", "recording", "telephoto",
         "ultrawide", "picture", "photos", "pictures", "selfies"),
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
         "budget", "affordable", "worth", "value", "pricey", "priciest",
         "priced"),
        None,  # prices live in their own table
        False,
    ),
    "design": (
        ("design", "build", "weight", "weigh", "weighs", "light", "lightest",
         "lighter", "heavy", "heavier", "heaviest",
         "dimensions", "thin", "color", "colours", "colors", "material",
         "waterproof", "ip68", "durable"),
        "weight_g",
        False,  # lighter is better
    ),
    "charging": (
        ("charge", "charges", "charging", "charging speed", "fast charging", "watt",
         "watts", "wattage", "w charging", "wireless charging"),
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

# Casual ways of asking what a phone costs that contain no price keyword.
# "how much does the S24 weigh" is deliberately not covered: only "how much
# is/are/for/would" and "set me back" are, and a competing aspect keyword
# still outranks the phrase.
_PRICE_PHRASES = re.compile(
    r"\bhow much (?:is|are|for|would|will)\b|\bset (?:me|you|us) back\b"
)

# Superlative direction. Without this "smallest battery" and "heaviest phone"
# returned the opposite end of the ranking, because the direction came only
# from the aspect's notion of "better". Minimum words are checked first, so
# "least expensive" is not read as "most". "best" and "top" are in neither
# list: they mean "best on this aspect" (lightest, for weight).
_MIN_WORDS = (
    "smallest", "lowest", "least", "minimum", "shortest", "fewest",
    "cheapest", "lightest", "slowest", "tiniest",
)
_MAX_WORDS = (
    "largest", "biggest", "highest", "most", "maximum", "longest", "greatest",
    "heaviest", "priciest", "fastest", "strongest",
)

# Metrics with their own column, which the aspect table cannot express since
# each aspect ranks by one headline column ("selfie camera" is not the main
# camera; "most RAM" is not the newest phone).
_COLUMN_OVERRIDES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bselfies?\b|\bfront(?:-facing)? camera\b"), "selfie_camera_mp"),
    (re.compile(r"\bram\b"), "max_ram_gb"),
)

# Price rankings default to EUR, the currency every listing has.
_CURRENCIES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\busd\b|\bdollars?\b|\$"), "USD"),
    (re.compile(r"\bgbp\b|\bpounds?\b|\u00a3"), "GBP"),
    (re.compile(r"\binr\b|\brupees?\b|\u20b9"), "INR"),
)

# "Which phone should I buy for storage?" contains the superlative trigger
# "which phone" but is a recommendation.
_STRONG_RECOMMENDATION_WORDS = ("should i buy", "should i get", "recommend")

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
    currency: str = "EUR"  # only used when ranking by price

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

    if _PRICE_PHRASES.search(lowered):
        # Worth one keyword, so "how much is the S24's battery" still ranks
        # battery first (an earlier entry in ASPECTS wins a tie).
        hits = [(s + 1, a) if a == "price" else (s, a) for s, a in hits]
        if not any(a == "price" for _, a in hits):
            hits.append((1, "price"))

    hits.sort(key=lambda item: item[0], reverse=True)
    return [aspect for _, aspect in hits]


# --------------------------------------------------------------------------
# Model-name normalisation
# --------------------------------------------------------------------------
_NAME_REWRITES: tuple[tuple[re.Pattern, str], ...] = (
    # "S-22", "A 54" -> "s22", "a54"
    (re.compile(r"\b([a-z])[-\s](\d{2})\b"), r"\1\2"),
    # "s23ultra", "s25plus", "s23fe" -> "s23 ultra", ...
    (re.compile(r"\b([a-z]\d{2})(ultra|plus|fe)\b"), r"\1 \2"),
    # "S23U", "S23 U" -> "s23 ultra"
    (re.compile(r"\b(s\d{2})\s?u\b"), r"\1 ultra"),
    (re.compile(r"\bfan edition\b"), "fe"),
    # "Fold 5", "Z Flip 5", "fold5" -> "z fold5": the catalogue spells them
    # "Z Fold5", and the Z is usually dropped in speech.
    (re.compile(r"\b(?:z\s?)?(fold|flip)\s?(\d)\b"), r"z \1\2"),
)


def normalize_model_names(query: str, vocabulary: set[str] | None = None) -> str:
    """Rewrite nicknames and spelling variants of model names canonically.

    The database lookup needs every token of a model name in the question, so
    "S23U", "Fold 5" or "Galaxy S-22" matched nothing and the answer fell back
    to an unfiltered search. The result is only used to resolve phones; routing
    still reads the original question.

    `vocabulary` (tokens of the catalogue's model names) enables typo
    correction: a misspelt word such as "Ultar" becomes the vocabulary word it
    most resembles. Only alphabetic words of four or more letters are touched,
    at a 0.8 similarity cutoff, so ordinary words are left alone.
    """
    text = query.lower()
    for pattern, replacement in _NAME_REWRITES:
        text = pattern.sub(replacement, text)

    if vocabulary:
        candidates = sorted(w for w in vocabulary if w.isalpha() and len(w) >= 4)

        def _fix(match: re.Match) -> str:
            word = match.group(0)
            close = difflib.get_close_matches(word, candidates, n=1, cutoff=0.8)
            return close[0] if close else word

        text = re.sub(r"\b[a-z]{4,}\b", _fix, text)
    return text


def _ranking_direction(lowered: str, default: bool) -> bool:
    """True when a superlative asks for the highest value of its metric."""
    if any(re.search(rf"\b{word}\b", lowered) for word in _MIN_WORDS):
        return False
    if any(re.search(rf"\b{word}\b", lowered) for word in _MAX_WORDS):
        return True
    return default


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
        for pattern, column in _COLUMN_OVERRIDES:
            if pattern.search(lowered):
                ranking_column = column
                break
        higher_is_better = _ranking_direction(lowered, higher_is_better)

    currency = "EUR"
    for pattern, code in _CURRENCIES:
        if pattern.search(lowered):
            currency = code
            break

    is_superlative = any(word in lowered for word in _SUPERLATIVE_WORDS)
    is_comparison = any(word in lowered for word in _COMPARISON_WORDS)

    if mentioned_phone_count >= 2:
        intent = Intent.COMPARISON
    elif is_comparison and mentioned_phone_count >= 1:
        intent = Intent.COMPARISON
    elif any(word in lowered for word in _LIST_WORDS):
        intent = Intent.LIST
    elif (
        any(word in lowered for word in _STRONG_RECOMMENDATION_WORDS)
        and mentioned_phone_count == 0
    ):
        intent = Intent.RECOMMENDATION
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
        currency=currency,
    )
