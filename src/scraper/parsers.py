"""Turn GSMArena's free-text spec strings into typed values.

GSMArena writes specs for humans ("Li-Ion 5000 mAh, non-removable"), so every
numeric column in the schema comes from a small parser here. Each one returns
`None` rather than guessing when the text does not contain the value, which
keeps unparseable entries out of the rankings instead of poisoning them.
"""

from __future__ import annotations

import re

CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "¥": "JPY",
    "C$": "CAD",
    "A$": "AUD",
}


def clean_text(value: str | None) -> str | None:
    """Collapse whitespace and strip the separators GSMArena pads cells with."""
    if not value:
        return None
    #   (thin space) and \xa0 (nbsp) appear throughout GSMArena markup.
    text = value.replace(" ", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("-–—•· ").strip()
    return text or None


def parse_battery_mah(value: str | None) -> int | None:
    """`"Li-Ion 5000 mAh, non-removable"` -> `5000`."""
    if not value:
        return None
    match = re.search(r"(\d{3,6})\s*mAh", value, re.IGNORECASE)
    if not match:
        return None
    capacity = int(match.group(1))
    # Guard against stray matches; phone batteries live in this range.
    return capacity if 500 <= capacity <= 12000 else None


def parse_display_inches(value: str | None) -> float | None:
    """`"6.1 inches, 90.1 cm2 (~86.8% screen-to-body ratio)"` -> `6.1`."""
    if not value:
        return None
    match = re.search(r"(\d+\.?\d*)\s*inch", value, re.IGNORECASE)
    if not match:
        return None
    inches = float(match.group(1))
    return inches if 2.0 <= inches <= 15.0 else None


def parse_refresh_rate(value: str | None) -> int | None:
    """Pull the highest advertised refresh rate out of a display type string."""
    if not value:
        return None
    rates = [int(hz) for hz in re.findall(r"(\d{2,3})\s*Hz", value, re.IGNORECASE)]
    valid = [hz for hz in rates if 30 <= hz <= 240]
    return max(valid) if valid else None


def parse_weight_grams(value: str | None) -> float | None:
    """`"168 g (5.93 oz)"` -> `168.0`."""
    if not value:
        return None
    match = re.search(r"(\d+\.?\d*)\s*g\b", value)
    if not match:
        return None
    grams = float(match.group(1))
    return grams if 50 <= grams <= 800 else None


def parse_charging_watts(value: str | None) -> float | None:
    """Highest wired/wireless wattage mentioned in the charging string."""
    if not value:
        return None
    watts = [float(w) for w in re.findall(r"(\d+\.?\d*)\s*W\b", value)]
    valid = [w for w in watts if 1 <= w <= 400]
    return max(valid) if valid else None


def parse_camera_mp(value: str | None) -> float | None:
    """Resolution of the primary sensor — the first one GSMArena lists.

    Taking the first rather than the maximum matters: on the S23 Ultra the list
    reads "200 MP wide, 10 MP periscope, 10 MP telephoto, 12 MP ultrawide", and
    the headline sensor is the one that leads.
    """
    if not value:
        return None
    match = re.search(r"(\d+\.?\d*)\s*MP", value, re.IGNORECASE)
    if not match:
        return None
    megapixels = float(match.group(1))
    return megapixels if 0.3 <= megapixels <= 500 else None


def parse_memory_options(value: str | None) -> tuple[int | None, int | None]:
    """`"128GB 8GB RAM, 256GB 12GB RAM"` -> `(max_ram_gb, max_storage_gb)`.

    Storage and RAM share the same `NNGB` shape, so they are told apart by the
    `RAM` suffix: any figure followed by "RAM" is memory, everything else is
    storage.
    """
    if not value:
        return None, None

    ram_values = [
        int(gb) for gb in re.findall(r"(\d+)\s*GB\s*RAM", value, re.IGNORECASE)
    ]
    storage_values = [
        int(gb)
        for gb in re.findall(r"(\d+)\s*GB(?!\s*RAM)", value, re.IGNORECASE)
        if int(gb) >= 8
    ]
    # Terabyte variants exist on Ultra models.
    storage_values += [
        int(float(tb) * 1024) for tb in re.findall(r"(\d+)\s*TB", value, re.IGNORECASE)
    ]

    max_ram = max(ram_values) if ram_values else None
    # Anything also claimed as RAM is not storage.
    storage_only = [s for s in storage_values if s not in set(ram_values)]
    max_storage = max(storage_only) if storage_only else None
    return max_ram, max_storage


def parse_release_year(value: str | None) -> int | None:
    """`"2023, February 01"` -> `2023`."""
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    if not match:
        return None
    year = int(match.group(0))
    return year if 1995 <= year <= 2100 else None


def parse_prices(value: str | None) -> list[dict]:
    """Split GSMArena's combined price cell into one row per currency.

    A typical cell reads `"$ 599.99 / € 649.00 / £ 559.00 / ₹ 74,999"`.
    """
    if not value:
        return []

    prices: list[dict] = []
    seen: set[str] = set()

    pattern = re.compile(r"([$€£₹¥]|C\$|A\$)\s*([\d,]+(?:\.\d{1,2})?)")
    for symbol, raw_amount in pattern.findall(value):
        currency = CURRENCY_SYMBOLS.get(symbol)
        if not currency or currency in seen:
            continue
        try:
            amount = float(raw_amount.replace(",", ""))
        except ValueError:
            continue
        if amount <= 0:
            continue
        seen.add(currency)
        prices.append(
            {
                "currency": currency,
                "amount": amount,
                "raw_text": f"{symbol} {raw_amount}",
                "source": "GSMArena",
            }
        )

    return prices
