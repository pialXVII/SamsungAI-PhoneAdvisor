"""Generate the evaluation set `eval/dataset.jsonl` from the phone snapshot.

    python -m eval.build_dataset            # writes eval/dataset.jsonl
    python -m eval.build_dataset --check    # rebuild in memory, diff against the file

Every gold label is computed from the database rows loaded out of
`data/scraped_phones.json`, never typed in by hand, and then re-verified by
`verify()` before anything is written:

- gold phones exist in the database;
- every gold fact occurs in that phone's own passage for the gold aspect (or in
  its price rows), so a correct system *can* find it;
- superlative and recommendation winners are recomputed with a plain Python
  sort that re-implements the tie-break rule of `repository.top_by_column`
  (metric, then newest `release_year`, then name A-Z), and must agree with it;
- intents and aspects are real names from `query_analysis.Intent` and
  `documents.py`.

Question wording comes from the hand-written templates below; which template,
which phone and which name form is used is drawn from a seeded RNG, so the
output is byte-identical from run to run. Splits are stratified: each category
is shuffled with the same seed and cut in half (dev / test).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from eval import _env

SEED = 20260928
EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.jsonl"

DOC_ASPECTS = (
    "Overview",
    "Display",
    "Camera",
    "Performance",
    "Battery and charging",
    "Design and build",
    "Connectivity and features",
    "Pricing",
)


# --------------------------------------------------------------------------
# Facts pulled from a phone row
# --------------------------------------------------------------------------
def _num(value: float | int) -> str:
    """Render a number the way the spec text does (50.0 -> "50")."""
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _money(value: float) -> str:
    return f"{value:.2f}"


def _chip(phone) -> str:
    """The first named chipset, e.g. "Snapdragon 8 Gen 2" or "Exynos 1380"."""
    match = re.search(r"(Snapdragon [0-9A-Za-z ]+?|Exynos \d+)\s*\(", phone.chipset or "")
    return match.group(1).strip() if match else (phone.chipset or "")


def _bluetooth(phone) -> str:
    return (phone.bluetooth or "").split(",")[0].strip()


def _storage(phone) -> str:
    gb = phone.max_storage_gb
    return f"{gb // 1024}TB" if gb and gb >= 1024 and gb % 1024 == 0 else f"{gb}GB"


def _price(phone, currency: str) -> float | None:
    amounts = [p.amount for p in phone.prices if p.currency == currency]
    return min(amounts) if amounts else None


# query-level aspect -> (document aspect, fact extractor)
FACTS = {
    "display": ("Display", lambda p: [_num(p.display_size_inches), str(p.display_refresh_rate_hz)]),
    "camera": ("Camera", lambda p: [_num(p.main_camera_mp), _num(p.selfie_camera_mp)]),
    "performance": ("Performance", lambda p: [_chip(p)]),
    "storage": ("Performance", lambda p: [_storage(p)]),
    "battery": ("Battery and charging", lambda p: [str(p.battery_capacity_mah)]),
    "charging": ("Battery and charging", lambda p: [f"{_num(p.charging_watts)}W"]),
    "design": ("Design and build", lambda p: [_num(p.weight_g)]),
    "connectivity": ("Connectivity and features", lambda p: [_bluetooth(p)]),
    "software": ("Performance", lambda p: [p.os.split(",")[0].strip()]),
    "overview": ("Overview", lambda p: [str(p.release_year), _chip(p)]),
    "price": ("Pricing", lambda p: [_money(_price(p, "EUR")), _money(_price(p, "USD"))]),
}

def _resolution(phone) -> str:
    """"1080 x 2340 pixels, 19.5:9 ratio (...)" -> "1080 x 2340"."""
    return (phone.display_resolution or "").split(" pixels")[0].strip()


def _frame(phone) -> str:
    """The frame material from the build line, e.g. "aluminum frame"."""
    match = re.search(r"(\w+ frame)", phone.build or "")
    return match.group(1) if match else (phone.build or "").split(",")[0].strip()


def _updates(phone) -> str:
    """The update promise after the Android version, e.g. "4 major OS updates"."""
    parts = [part.strip() for part in (phone.os or "").split(",")]
    return (parts[1] if len(parts) > 1 else parts[0]).removeprefix("up to ")


# Templates that ask for something narrower or different from the aspect's
# default facts. Without these, "What sensors and USB port...?" was graded on
# the Bluetooth version and "How many OS updates...?" on the Android version.
TEMPLATE_FACTS = {
    "What screen resolution and refresh rate does the {p} have?": lambda p: [
        _resolution(p), str(p.display_refresh_rate_hz)
    ],
    "How many megapixels is the {p} main camera?": lambda p: [_num(p.main_camera_mp)],
    "Tell me about the {p} design and build materials.": lambda p: [_frame(p)],
    "Does the {p} have NFC and Wi-Fi 6?": lambda p: ["NFC", (p.wlan or "").split(",")[0].strip()],
    "What sensors and USB port does the {p} have?": lambda p: [(p.usb or "").split(",")[0].strip()],
    "How many OS updates does the {p} get?": lambda p: [_updates(p)],
    "How big is the {p} screen?": lambda p: [_num(p.display_size_inches)],
    "How sharp is the display on the {p}?": lambda p: [_resolution(p)],
    "How much does the {p} cost in euros?": lambda p: [_money(_price(p, "EUR"))],
}


def _facts_for(template: str, aspect: str, phone) -> list[str]:
    extractor = TEMPLATE_FACTS.get(template) or FACTS[aspect][1]
    return extractor(phone)


# Document aspects that legitimately contain the answer, for retrieval scoring.
ACCEPTABLE_ASPECTS = {
    "Pricing": ["Pricing", "Overview"],
    "Overview": list(DOC_ASPECTS),
}


# --------------------------------------------------------------------------
# Name forms
# --------------------------------------------------------------------------
def canonical_forms(name: str) -> list[str]:
    """Unambiguous ways people write a model name: every model token kept."""
    base = name.removeprefix("Samsung ").removesuffix(" 5G")  # "Galaxy S21"
    short = base.removeprefix("Galaxy ")                      # "S21"
    return sorted({name, base, short, f"Samsung {short}"})


# Nicknames and sloppy spellings a real user types. Each maps to exactly one
# phone; none contains every token of the official name, which is the point.
NICKNAMES: dict[str, list[str]] = {
    "Samsung Galaxy S23 Ultra": ["S23U", "s23ultra", "Galaxy S23 Ultar"],
    "Samsung Galaxy S24 Ultra": ["S24U", "the s24ultra"],
    "Samsung Galaxy S25 Ultra": ["S25U", "Samsung S25ultra"],
    "Samsung Galaxy S22 Ultra 5G": ["S22U"],
    "Samsung Galaxy S21 Ultra 5G": ["S21U"],
    "Samsung Galaxy Z Fold5": ["Fold 5", "Galaxy Fold5"],
    "Samsung Galaxy Z Fold6": ["Fold 6", "Z Fold 6"],
    "Samsung Galaxy Z Flip5": ["Flip 5", "Galaxy Flip5"],
    "Samsung Galaxy A54": ["A-54", "Samsng A54"],
    "Samsung Galaxy S23 FE": ["S23 Fan Edition", "s23fe"],
    "Samsung Galaxy S24": ["Samsnug S24"],
    "Samsung Galaxy S22 5G": ["Galaxy S-22"],
}


# --------------------------------------------------------------------------
# Hand-written templates
# --------------------------------------------------------------------------
SPEC_TEMPLATES = {
    "display": [
        "What are the display specs of the {p}?",
        "Tell me about the {p} screen.",
        "What screen resolution and refresh rate does the {p} have?",
    ],
    "camera": [
        "What are the camera specs of the {p}?",
        "Tell me about the {p} camera setup.",
        "How many megapixels is the {p} main camera?",
    ],
    "performance": [
        "What processor does the {p} use?",
        "What chipset is in the {p}?",
        "Tell me about the {p} performance and CPU.",
    ],
    "storage": [
        "What storage options does the {p} come with?",
        "How much internal storage can I get on the {p}?",
    ],
    "battery": [
        "What is the battery capacity of the {p}?",
        "How many mAh is the {p} battery?",
        "Tell me about the {p} battery.",
    ],
    "charging": [
        "What charging speed does the {p} support?",
        "How fast is wired charging on the {p}?",
    ],
    "design": [
        "How much does the {p} weigh?",
        "What are the dimensions and weight of the {p}?",
        "Tell me about the {p} design and build materials.",
    ],
    "connectivity": [
        "What Bluetooth version does the {p} have?",
        "Does the {p} have NFC and Wi-Fi 6?",
        "What sensors and USB port does the {p} have?",
    ],
    "software": [
        "What Android version does the {p} ship with?",
        "How many OS updates does the {p} get?",
    ],
    "overview": [
        "Give me an overview of the {p}.",
        "Tell me about the {p}.",
    ],
}

PARAPHRASE_TEMPLATES = {
    "battery": ["How long does the {p} battery last?", "Is the battery on the {p} any good?"],
    "camera": ["How good are the photos from the {p}?", "Can the {p} take nice pictures?"],
    "display": ["How big is the {p} screen?", "How sharp is the display on the {p}?"],
    "design": ["Is the {p} heavy?", "How heavy is the {p}?"],
    "performance": ["Is the {p} fast enough for gaming?", "What chip powers the {p}?"],
    "price": ["How much does the {p} go for?", "What would the {p} set me back?"],
}

PRICE_TEMPLATES = [
    "How much is the {p}?",
    "What is the price of the {p}?",
    "What does the {p} cost?",
    "Cost of the {p}?",
    "How much does the {p} cost in euros?",
    "How expensive is the {p}?",
    "What's the {p} price?",
]

COMPARISON_TEMPLATES = {
    "camera": [
        "Compare the camera of the {a} and the {b}.",
        "{a} vs {b}: which has the better camera?",
        "Is the {a} camera better than the {b}?",
    ],
    "battery": [
        "Compare the battery life of the {a} and {b}.",
        "{a} or {b} for battery life?",
        "What's the difference between the {a} and {b} batteries?",
    ],
    "display": [
        "Compare the displays of the {a} and the {b}.",
        "{a} vs {b} screen size?",
    ],
    "performance": [
        "{a} or {b} for gaming?",
        "Compare the performance of the {a} and {b}.",
        "Which is faster, the {a} or the {b}?",
    ],
    "design": [
        "Is the {a} lighter than the {b}?",
        "Compare the weight of the {a} versus the {b}.",
    ],
}

# Fact used for each phone in a comparison (both phones' values are gold).
COMPARISON_FACT = {
    "camera": lambda p: _num(p.main_camera_mp),
    "battery": lambda p: str(p.battery_capacity_mah),
    "display": lambda p: _num(p.display_size_inches),
    "performance": _chip,
    "design": lambda p: _num(p.weight_g),
}

# Pairs people actually cross-shop: siblings, generations, foldables.
COMPARISON_PAIRS = [
    ("Samsung Galaxy S23", "Samsung Galaxy S22 5G"),
    ("Samsung Galaxy S24", "Samsung Galaxy S23"),
    ("Samsung Galaxy S25", "Samsung Galaxy S24"),
    ("Samsung Galaxy S23 Ultra", "Samsung Galaxy S22 Ultra 5G"),
    ("Samsung Galaxy S24 Ultra", "Samsung Galaxy S23 Ultra"),
    ("Samsung Galaxy S25 Ultra", "Samsung Galaxy S24 Ultra"),
    ("Samsung Galaxy S21 Ultra 5G", "Samsung Galaxy S25 Ultra"),
    ("Samsung Galaxy Z Fold6", "Samsung Galaxy Z Fold5"),
    ("Samsung Galaxy Z Flip5", "Samsung Galaxy Z Fold5"),
    ("Samsung Galaxy A54", "Samsung Galaxy S23 FE"),
    ("Samsung Galaxy S21 5G", "Samsung Galaxy S25"),
    ("Samsung Galaxy Z Fold6", "Samsung Galaxy S24 Ultra"),
    # Same-line pairs: one name is a prefix of the other.
    ("Samsung Galaxy S23", "Samsung Galaxy S23 Ultra"),
    ("Samsung Galaxy S24", "Samsung Galaxy S24 Ultra"),
    ("Samsung Galaxy S25", "Samsung Galaxy S25 Ultra"),
    ("Samsung Galaxy S23", "Samsung Galaxy S23 FE"),
]


@dataclass(frozen=True)
class Ranking:
    """A superlative the database can answer: column, direction, value getter."""

    column: str | None  # Phone column for top_by_column; None = special
    descending: bool
    key: str            # label used in facts / metadata


# (question, ranking, query aspect) — every question hand-written.
SUPERLATIVES = [
    ("Which Samsung phone has the biggest battery?", Ranking("battery_capacity_mah", True, "battery"), "battery"),
    ("What phone has the longest battery life?", Ranking("battery_capacity_mah", True, "battery"), "battery"),
    ("Which model has the highest mAh rating?", Ranking("battery_capacity_mah", True, "battery"), "battery"),
    ("Which phone has the smallest battery?", Ranking("battery_capacity_mah", False, "battery"), "battery"),
    ("Which phone has the highest megapixel main camera?", Ranking("main_camera_mp", True, "camera"), "camera"),
    ("Which Samsung has the best camera resolution?", Ranking("main_camera_mp", True, "camera"), "camera"),
    ("Which phone has the highest resolution selfie camera?", Ranking("selfie_camera_mp", True, "selfie"), "camera"),
    ("Which phone has the largest screen?", Ranking("display_size_inches", True, "display"), "display"),
    ("Which Samsung has the biggest display?", Ranking("display_size_inches", True, "display"), "display"),
    ("Which phone has the smallest display?", Ranking("display_size_inches", False, "display"), "display"),
    ("Which phone offers the most storage?", Ranking("max_storage_gb", True, "storage"), "storage"),
    ("Which phone has the most RAM?", Ranking("max_ram_gb", True, "ram"), "performance"),
    ("Which phone has the fastest charging?", Ranking("charging_watts", True, "charging"), "charging"),
    ("What is the highest wattage charging on any Samsung phone?", Ranking("charging_watts", True, "charging"), "charging"),
    ("What is the lightest Samsung phone?", Ranking("weight_g", False, "weight"), "design"),
    ("Which phone is the heaviest?", Ranking("weight_g", True, "weight"), "design"),
    ("What is the cheapest Samsung phone?", Ranking(None, False, "price_eur"), "price"),
    ("Which phone is the most expensive?", Ranking(None, True, "price_eur"), "price"),
    ("What is the cheapest phone in US dollars?", Ranking(None, False, "price_usd"), "price"),
    ("Which phone has the fastest processor?", Ranking("release_year", True, "generation"), "performance"),
    ("What is the most powerful Samsung phone for gaming?", Ranking("release_year", True, "generation"), "performance"),
]

# (question, ranking, query aspect); gold = every phone tied for best, except
# price where "budget" means the three cheapest EUR listings.
RECOMMENDATIONS = [
    ("Which Samsung phone should I buy for photography?", Ranking("main_camera_mp", True, "camera"), "camera"),
    ("I take a lot of photos. What do you recommend?", Ranking("main_camera_mp", True, "camera"), "camera"),
    ("Recommend a Samsung phone for selfies.", Ranking("selfie_camera_mp", True, "selfie"), "camera"),
    ("I want great battery life, what do you recommend?", Ranking("battery_capacity_mah", True, "battery"), "battery"),
    ("Which phone should I buy if I travel and need a battery that lasts?", Ranking("battery_capacity_mah", True, "battery"), "battery"),
    ("Recommend a Samsung phone for gaming.", Ranking("release_year", True, "generation"), "performance"),
    ("I need a powerful phone for heavy apps, which one should I get?", Ranking("release_year", True, "generation"), "performance"),
    ("Can you recommend a budget Samsung phone?", Ranking(None, False, "price_eur"), "price"),
    ("What's an affordable phone you would recommend for a student?", Ranking(None, False, "price_eur"), "price"),
    ("Recommend a phone with a big screen for reading.", Ranking("display_size_inches", True, "display"), "display"),
    ("Which one should I buy for watching movies on a large display?", Ranking("display_size_inches", True, "display"), "display"),
    ("Recommend a light phone that is easy to carry.", Ranking("weight_g", False, "weight"), "design"),
    ("I store lots of videos, which phone should I buy for storage?", Ranking("max_storage_gb", True, "storage"), "storage"),
    ("Recommend a phone with fast charging.", Ranking("charging_watts", True, "charging"), "charging"),
]

OUT_OF_DOMAIN = [
    ("other_brand", "What are the camera specs of the iPhone 15 Pro?"),
    ("other_brand", "How big is the Google Pixel 8 battery?"),
    ("other_brand", "How much is the OnePlus 12?"),
    ("other_brand", "Compare the iPhone 15 and the Pixel 8."),
    ("other_brand", "What processor does the Xiaomi 14 Ultra use?"),
    ("other_brand", "Is the Nothing Phone 2 good for gaming?"),
    ("other_brand", "What display does the Motorola Edge 40 have?"),
    ("other_brand", "Which is better for photos, Pixel 8 Pro or iPhone 15 Pro Max?"),
    ("unknown_samsung", "What is the battery capacity of the Galaxy S20?"),
    ("unknown_samsung", "Tell me about the Galaxy Note 20 Ultra camera."),
    ("unknown_samsung", "How much is the Galaxy A15?"),
    ("unknown_samsung", "What chipset does the Galaxy S10 have?"),
    ("unknown_samsung", "What is the battery life of the Galaxy Watch 6?"),
    ("unknown_samsung", "How much does a Samsung Galaxy Tab S9 cost?"),
    ("unrelated", "What is the best laptop for programming?"),
    ("unrelated", "How do I bake sourdough bread?"),
    ("unrelated", "What is the capital of Australia?"),
    ("unrelated", "Recommend a good Samsung refrigerator."),
    ("unrelated", "Who won the 2022 FIFA World Cup?"),
    ("unrelated", "Write a Python function that reverses a string."),
    ("unrelated", "How much does a Tesla Model 3 cost?"),
    ("unrelated", "What's the weather like in Dhaka today?"),
    ("chitchat", "Hi, how are you?"),
    ("chitchat", "Tell me a joke."),
    ("chitchat", "Thanks, that's all!"),
    ("chitchat", "What is your name?"),
    ("chitchat", "Can you sing me a song?"),
    ("chitchat", "I'm bored."),
    ("chitchat", "Good morning!"),
    ("chitchat", "What is the meaning of life?"),
]


# --------------------------------------------------------------------------
# Gold computation, independent of SQL
# --------------------------------------------------------------------------
def _metric(phone, ranking: Ranking):
    if ranking.key == "price_eur":
        return _price(phone, "EUR")
    if ranking.key == "price_usd":
        return _price(phone, "USD")
    return getattr(phone, ranking.column)


def rank_phones(phones, ranking: Ranking) -> list:
    """Best first, ties by newest release_year, then name A-Z.

    Mirrors `repository.top_by_column`: the metric in the requested direction,
    then `release_year` descending, then `name` ascending. Prices have no tie in
    the snapshot, so the same rule is used for them.
    """
    usable = [p for p in phones if _metric(p, ranking) is not None]
    sign = -1 if ranking.descending else 1
    return sorted(
        usable, key=lambda p: (sign * _metric(p, ranking), -(p.release_year or 0), p.name)
    )


def _metric_fact(phone, ranking: Ranking) -> str:
    value = _metric(phone, ranking)
    if ranking.key.startswith("price"):
        return _money(value)
    if ranking.key == "generation":
        return _chip(phone)
    return _num(value)


def tied_best(phones, ranking: Ranking) -> list:
    ranked = rank_phones(phones, ranking)
    top = _metric(ranked[0], ranking)
    return [p for p in ranked if _metric(p, ranking) == top]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def _item(category, question, intent, phones=(), aspect=None, facts=(), **extra) -> dict:
    record = {
        "id": None,
        "split": None,
        "category": category,
        "question": question,
        "gold_intent": intent,
        "gold_phones": list(phones),
        "gold_aspect": aspect,
        "gold_aspects_ok": ACCEPTABLE_ASPECTS.get(aspect, [aspect]) if aspect else [],
        "gold_answer_facts": list(facts),
        "expect_decline": False,
    }
    record.update(extra)
    return record


def generate(phones) -> list[dict]:
    rng = random.Random(SEED)
    by_name = {p.name: p for p in phones}
    names = sorted(by_name)
    items: list[dict] = []

    # spec_lookup: every phone x a seeded sample of aspects, all aspects covered.
    aspects = list(SPEC_TEMPLATES)
    for index, name in enumerate(names):
        phone = by_name[name]
        # Rotate so each aspect appears for several phones, then add one random.
        chosen = {aspects[(index * 3 + k) % len(aspects)] for k in range(3)}
        for aspect in sorted(chosen):
            doc_aspect = FACTS[aspect][0]
            # Same RNG call order as before: template first, then name form.
            template = rng.choice(SPEC_TEMPLATES[aspect])
            items.append(
                _item(
                    "spec_lookup",
                    template.format(p=rng.choice(canonical_forms(name))),
                    "spec_lookup",
                    [name],
                    doc_aspect,
                    _facts_for(template, aspect, phone),
                    query_aspect=aspect,
                )
            )

    # price: every phone once.
    for name in names:
        template = rng.choice(PRICE_TEMPLATES)
        items.append(
            _item(
                "price",
                template.format(p=rng.choice(canonical_forms(name))),
                "price",
                [name],
                "Pricing",
                _facts_for(template, "price", by_name[name]),
                query_aspect="price",
            )
        )

    # comparison: each pair with two different aspects.
    comp_aspects = list(COMPARISON_TEMPLATES)
    for a, b in COMPARISON_PAIRS:
        for aspect in rng.sample(comp_aspects, 2):
            pa, pb = by_name[a], by_name[b]
            if rng.random() < 0.5:
                pa, pb = pb, pa
            question = rng.choice(COMPARISON_TEMPLATES[aspect]).format(
                a=rng.choice(canonical_forms(pa.name)), b=rng.choice(canonical_forms(pb.name))
            )
            items.append(
                _item(
                    "comparison",
                    question,
                    "comparison",
                    [pa.name, pb.name],
                    None,
                    [COMPARISON_FACT[aspect](pa), COMPARISON_FACT[aspect](pb)],
                    query_aspect=aspect,
                )
            )

    # superlative
    for question, ranking, aspect in SUPERLATIVES:
        ranked = rank_phones(phones, ranking)
        tied = tied_best(phones, ranking)
        items.append(
            _item(
                "superlative",
                question,
                "superlative",
                [ranked[0].name],
                None,
                [_metric_fact(ranked[0], ranking)],
                query_aspect=aspect,
                ranking={"column": ranking.column, "descending": ranking.descending, "key": ranking.key},
                gold_tied_phones=[p.name for p in tied],
            )
        )

    # recommendation
    for question, ranking, aspect in RECOMMENDATIONS:
        if ranking.key.startswith("price"):
            gold = rank_phones(phones, ranking)[:3]
        else:
            gold = tied_best(phones, ranking)
        items.append(
            _item(
                "recommendation",
                question,
                "recommendation",
                [p.name for p in gold],
                None,
                [],
                query_aspect=aspect,
                ranking={"column": ranking.column, "descending": ranking.descending, "key": ranking.key},
            )
        )

    # paraphrase / nickname: nickname x paraphrased wording, plus canonical
    # names with paraphrased wording.
    para_aspects = list(PARAPHRASE_TEMPLATES)
    nick_pairs = [(name, nick) for name in sorted(NICKNAMES) for nick in NICKNAMES[name]]
    for name, nick in nick_pairs:
        aspect = rng.choice(para_aspects)
        items.append(_paraphrase(rng, by_name[name], nick, aspect, "nickname"))
    for name in rng.sample(names, 10):
        aspect = rng.choice(para_aspects)
        items.append(
            _paraphrase(rng, by_name[name], rng.choice(canonical_forms(name)), aspect, "wording")
        )

    # out_of_domain
    for subtype, question in OUT_OF_DOMAIN:
        items.append(
            _item("out_of_domain", question, "general", expect_decline=True, ood_type=subtype)
        )

    return _assign_splits(rng, items)


def _paraphrase(rng, phone, form: str, aspect: str, variant: str) -> dict:
    doc_aspect = FACTS[aspect][0]
    template = rng.choice(PARAPHRASE_TEMPLATES[aspect])
    return _item(
        "paraphrase",
        template.format(p=form),
        "price" if aspect == "price" else "spec_lookup",
        [phone.name],
        doc_aspect,
        _facts_for(template, aspect, phone),
        query_aspect=aspect,
        variant=variant,
    )


def _assign_splits(rng, items: list[dict]) -> list[dict]:
    """Stratified 50/50 dev/test per category; ids are stable per category."""
    out: list[dict] = []
    categories = sorted({item["category"] for item in items})
    for category in categories:
        group = [item for item in items if item["category"] == category]
        for number, item in enumerate(group, start=1):
            item["id"] = f"{category}-{number:03d}"
        order = list(range(len(group)))
        rng.shuffle(order)
        dev = set(order[: (len(group) + 1) // 2])
        for index, item in enumerate(group):
            item["split"] = "dev" if index in dev else "test"
        out.extend(group)
    return out


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(text.replace(",", ""))}


def fact_in_text(fact: str, text: str) -> bool:
    """Numeric facts match as numbers; everything else case-insensitively."""
    if re.fullmatch(r"\d+(?:\.\d+)?", fact):
        return float(fact) in _numbers(text)
    return fact.lower() in text.lower()


def verify(items: list[dict], phones) -> None:
    from src.database.db import session_scope
    from src.database.repository import cheapest_phones, top_by_column
    from src.rag.documents import build_documents_for_phone
    from src.rag.query_analysis import Intent

    by_name = {p.name: p for p in phones}
    intents = {i.value for i in Intent}
    problems: list[str] = []

    def check(cond: bool, item: dict, message: str) -> None:
        if not cond:
            problems.append(f"{item['id']}: {message}")

    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert len({i["question"] for i in items}) == len(items), "duplicate questions"

    for item in items:
        check(item["gold_intent"] in intents, item, f"unknown intent {item['gold_intent']}")
        for aspect in [item["gold_aspect"], *item["gold_aspects_ok"]]:
            check(aspect is None or aspect in DOC_ASPECTS, item, f"unknown aspect {aspect}")
        for name in item["gold_phones"] + item.get("gold_tied_phones", []):
            check(name in by_name, item, f"unknown phone {name}")
        if item["expect_decline"]:
            check(not item["gold_phones"] and not item["gold_answer_facts"], item, "OOD with gold")
            continue

        if item["gold_aspect"]:
            # Single-phone lookup: each fact must be in that phone's gold passage.
            phone = by_name[item["gold_phones"][0]]
            passage = next(
                (d.text for d in build_documents_for_phone(phone) if d.aspect == item["gold_aspect"]),
                "",
            )
            for fact in item["gold_answer_facts"]:
                check(fact_in_text(fact, passage), item, f"fact {fact!r} not in {item['gold_aspect']}")
        elif item["category"] == "comparison":
            for name, fact in zip(item["gold_phones"], item["gold_answer_facts"]):
                check(fact_in_text(fact, by_name[name].spec_summary()), item, f"fact {fact!r} not in spec sheet")

        if item["category"] in ("superlative", "recommendation"):
            r = item["ranking"]
            ranking = Ranking(r["column"], r["descending"], r["key"])
            ranked = rank_phones(phones, ranking)
            with session_scope() as session:
                if r["key"] == "price_eur" and not r["descending"]:
                    sql = [p.name for p in cheapest_phones(session, limit=15)]
                elif r["key"].startswith("price"):
                    sql = None  # no repository helper for these; Python sort only
                else:
                    sql = [
                        p.name
                        for p in top_by_column(session, r["column"], limit=15, descending=r["descending"])
                    ]
            if sql is not None:
                check(sql == [p.name for p in ranked], item, "Python ranking disagrees with SQL")
            if item["category"] == "superlative":
                check(item["gold_phones"] == [ranked[0].name], item, "winner mismatch")
                check(fact_in_text(item["gold_answer_facts"][0], _metric_fact(ranked[0], ranking)), item, "fact")

    if problems:
        raise SystemExit("Gold verification failed:\n  " + "\n  ".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build eval/dataset.jsonl")
    parser.add_argument("--out", type=Path, default=DATASET_PATH)
    parser.add_argument("--check", action="store_true", help="Verify the file is up to date")
    args = parser.parse_args()

    _env.seed_database()
    from src.database.db import session_scope
    from src.database.repository import get_all_phones

    with session_scope() as session:
        phones = get_all_phones(session)
        items = generate(phones)
        verify(items, phones)

    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        print("dataset is up to date" if current == text else "dataset differs from generator")
        return 0 if current == text else 1

    args.out.write_text(text, encoding="utf-8")
    counts: dict[str, dict[str, int]] = {}
    for item in items:
        counts.setdefault(item["category"], {"dev": 0, "test": 0})[item["split"]] += 1
    print(f"Wrote {len(items)} questions to {args.out}")
    for category, split in sorted(counts.items()):
        print(f"  {category:15s} dev={split['dev']:3d} test={split['test']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
