"""Query helpers over the phone schema.

This is the single place that knows how to turn a loose user phrase like
"s23 ultra" into a row, so the RAG retriever, the agents and the API all resolve
model names identically.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Phone, Price, Specification

# Words that carry no signal when matching a model name.
_NOISE = {"samsung", "galaxy", "the", "phone", "smartphone", "5g", "4g", "lte"}


def _normalize(text: str) -> str:
    text = text.lower().replace("+", " plus ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(w for w in text.split() if w not in _NOISE)


def _token_set(text: str) -> set[str]:
    return set(_normalize(text).split())


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------
def upsert_phone(
    session: Session,
    phone_data: dict,
    raw_specs: Sequence[dict],
    prices: Sequence[dict],
) -> Phone:
    """Insert a phone or refresh it in place, replacing its specs and prices."""
    phone = session.scalar(select(Phone).where(Phone.name == phone_data["name"]))

    if phone is None:
        phone = Phone(**phone_data)
        session.add(phone)
    else:
        for key, value in phone_data.items():
            setattr(phone, key, value)
        # Children are fully rebuilt so a re-scrape never leaves stale rows.
        phone.specifications.clear()
        phone.prices.clear()
        session.flush()

    session.flush()

    seen: set[tuple[str, str]] = set()
    for spec in raw_specs:
        identity = (spec["category"], spec["name"])
        if identity in seen:
            continue
        seen.add(identity)
        phone.specifications.append(
            Specification(
                category=spec["category"], name=spec["name"], value=spec["value"]
            )
        )

    for price in prices:
        phone.prices.append(Price(**price))

    return phone


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------
def get_all_phones(session: Session) -> list[Phone]:
    return list(session.scalars(select(Phone).order_by(Phone.name)).all())


def count_phones(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Phone)) or 0


def get_phone_by_id(session: Session, phone_id: int) -> Phone | None:
    return session.get(Phone, phone_id)


def find_phone(session: Session, query: str) -> Phone | None:
    """Resolve a free-text model reference to exactly one phone.

    Tries exact match, then substring, then token-overlap, then fuzzy ratio.
    Token overlap is what makes "how good is the s23 ultra camera" land on
    *Galaxy S23 Ultra* rather than plain *Galaxy S23* — the extra token has to
    be accounted for, not merely tolerated.
    """
    phones = get_all_phones(session)
    if not phones:
        return None

    q_norm = _normalize(query)
    if not q_norm:
        return None

    for phone in phones:
        if _normalize(phone.name) == q_norm:
            return phone

    q_tokens = _token_set(query)
    best: tuple[float, Phone] | None = None

    for phone in phones:
        p_tokens = _token_set(phone.name)
        if not p_tokens:
            continue

        # Every token of the model name must appear in the query for a phone to
        # be a candidate; otherwise "s23" would match "s23 ultra" equally well.
        covered = len(p_tokens & q_tokens) / len(p_tokens)
        ratio = difflib.SequenceMatcher(None, q_norm, _normalize(phone.name)).ratio()

        # Longer names win ties so the more specific model is preferred.
        score = covered * 2.0 + ratio + (0.01 * len(p_tokens) if covered == 1.0 else 0)

        if covered < 1.0 and ratio < 0.72:
            continue
        if best is None or score > best[0]:
            best = (score, phone)

    return best[1] if best else None


def find_phones(session: Session, query: str, limit: int = 3) -> list[Phone]:
    """Return several plausible matches, best first (used for comparisons)."""
    phones = get_all_phones(session)
    q_tokens = _token_set(query)
    q_norm = _normalize(query)

    scored: list[tuple[float, Phone]] = []
    for phone in phones:
        p_tokens = _token_set(phone.name)
        if not p_tokens:
            continue
        covered = len(p_tokens & q_tokens) / len(p_tokens)
        ratio = difflib.SequenceMatcher(None, q_norm, _normalize(phone.name)).ratio()
        score = covered * 2.0 + ratio
        if covered > 0 or ratio > 0.5:
            scored.append((score, phone))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [phone for _, phone in scored[:limit]]


def extract_mentioned_phones(session: Session, query: str, limit: int = 4) -> list[Phone]:
    """Find every phone explicitly named in a sentence.

    Comparison questions ("how does the S23 compare to the S22") mention two
    models in one string, so a single best-match lookup is not enough.
    """
    phones = get_all_phones(session)
    q_tokens = _token_set(query)
    matches: list[tuple[float, Phone]] = []

    for phone in phones:
        p_tokens = _token_set(phone.name)
        if not p_tokens:
            continue
        # A phone counts as "mentioned" only when the query contains all of its
        # distinguishing tokens.
        if p_tokens.issubset(q_tokens):
            matches.append((len(p_tokens), phone))

    matches.sort(key=lambda item: item[0], reverse=True)

    # Drop phones whose tokens are fully contained in an already-selected, more
    # specific match, so "S23 Ultra" does not also drag in "S23".
    selected: list[Phone] = []
    for _, phone in matches:
        tokens = _token_set(phone.name)
        if any(tokens < _token_set(chosen.name) for chosen in selected):
            continue
        selected.append(phone)
        if len(selected) >= limit:
            break

    return selected


def top_by_column(
    session: Session, column_name: str, limit: int = 5, descending: bool = True
) -> list[Phone]:
    """Rank phones by any numeric column (battery, camera MP, display size).

    Ties are broken by release year, newest first. Six models share a 5000 mAh
    battery, so without a secondary sort "which has the best battery life?"
    would name whichever row the database happened to return first, and the
    answer could change between runs.
    """
    column = getattr(Phone, column_name, None)
    if column is None:
        raise ValueError(f"Unknown ranking column: {column_name}")
    order = column.desc() if descending else column.asc()
    stmt = (
        select(Phone)
        .where(column.isnot(None))
        .order_by(order, Phone.release_year.desc(), Phone.name.asc())
        .limit(limit)
    )
    return list(session.scalars(stmt).all())


def cheapest_phones(session: Session, limit: int = 5) -> list[Phone]:
    """Phones ordered by their lowest EUR listing."""
    stmt = (
        select(Phone, func.min(Price.amount).label("min_price"))
        .join(Price, Price.phone_id == Phone.id)
        .where(Price.currency == "EUR")
        .group_by(Phone.id)
        .order_by(func.min(Price.amount).asc())
        .limit(limit)
    )
    return [row[0] for row in session.execute(stmt).all()]


def search_specifications(session: Session, keyword: str, limit: int = 20) -> list[Specification]:
    pattern = f"%{keyword}%"
    stmt = (
        select(Specification)
        .where(Specification.value.ilike(pattern) | Specification.name.ilike(pattern))
        .limit(limit)
    )
    return list(session.scalars(stmt).all())


def database_stats(session: Session) -> dict:
    return {
        "phones": count_phones(session),
        "specifications": session.scalar(
            select(func.count()).select_from(Specification)
        )
        or 0,
        "prices": session.scalar(select(func.count()).select_from(Price)) or 0,
    }
