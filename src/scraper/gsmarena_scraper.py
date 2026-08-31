"""GSMArena scraper for Samsung handsets (requests + BeautifulSoup).

Two stages:

1. **Discovery** — walk the paginated Samsung brand listing
   (`samsung-phones-9.php` -> `samsung-phones-f-9-0-pN.php`) building a
   name -> URL index, stopping as soon as every target model is found.
2. **Extraction** — for each matched phone, parse `#specs-list`, which
   GSMArena renders as one `<table>` per category with `td.ttl` / `td.nfo`
   label/value pairs.

The crawl is deliberately polite: a configurable delay between requests,
retries with backoff on transient failures, and a single reused TCP session.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import config

from .parsers import (
    clean_text,
    parse_battery_mah,
    parse_camera_mp,
    parse_charging_watts,
    parse_display_inches,
    parse_memory_options,
    parse_prices,
    parse_refresh_rate,
    parse_release_year,
    parse_weight_grams,
)

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPhone:
    """Everything pulled from one phone page, ready for the repository."""

    phone: dict
    raw_specs: list[dict] = field(default_factory=list)
    prices: list[dict] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.phone["name"]


def _normalize_name(name: str) -> str:
    """Loose key for matching listing entries against target model names."""
    name = name.lower().replace("+", " plus ")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    # "5G" and "Galaxy" are noise: the listing writes "Galaxy S21 5G" while a
    # user (and our target list) may write either form.
    drop = {"samsung", "galaxy", "5g", "4g", "lte"}
    return " ".join(w for w in name.split() if w not in drop)


class GSMArenaScraper:
    """Scrapes Samsung phone specifications from GSMArena."""

    def __init__(
        self,
        delay: float | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.delay = config.SCRAPER_DELAY_SECONDS if delay is None else delay
        self.timeout = config.SCRAPER_TIMEOUT if timeout is None else timeout
        self.max_retries = (
            config.SCRAPER_MAX_RETRIES if max_retries is None else max_retries
        )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": config.SCRAPER_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }
        )
        self._last_request_at = 0.0

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_at = time.time()

    def fetch(self, url: str) -> BeautifulSoup | None:
        """GET a page with throttling and retries, returning parsed soup."""
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return BeautifulSoup(response.text, "html.parser")

                # 429/503 mean we are being rate-limited; back off harder.
                if response.status_code in (429, 503):
                    backoff = self.delay * (2**attempt)
                    logger.warning(
                        "Rate limited (%s) on %s, backing off %.1fs",
                        response.status_code,
                        url,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue

                logger.warning("HTTP %s for %s", response.status_code, url)
                return None
            except requests.RequestException as exc:
                logger.warning(
                    "Request failed (attempt %s/%s) for %s: %s",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                )
                time.sleep(self.delay * attempt)

        logger.error("Giving up on %s", url)
        return None

    # ------------------------------------------------------------------
    # Stage 1: discovery
    # ------------------------------------------------------------------
    def _listing_page_url(self, page: int) -> str:
        if page == 1:
            return urljoin(config.GSMARENA_BASE_URL, config.GSMARENA_SAMSUNG_LISTING)
        return urljoin(config.GSMARENA_BASE_URL, f"samsung-phones-f-9-0-p{page}.php")

    def parse_listing_page(self, soup: BeautifulSoup) -> dict[str, str]:
        """Extract `{display name: absolute url}` from one brand listing page."""
        found: dict[str, str] = {}
        for item in soup.select("div.makers ul li"):
            link = item.find("a")
            if not link or not link.get("href"):
                continue
            label = link.find("span")
            name = clean_text(label.get_text(" ", strip=True)) if label else None
            if not name:
                continue
            found[name] = urljoin(config.GSMARENA_BASE_URL, link["href"])
        return found

    def discover_phone_urls(
        self, targets: Iterable[str], max_pages: int | None = None
    ) -> dict[str, str]:
        """Walk listing pages until every target model has a URL.

        Returns `{target name: url}`. Missing targets are simply absent, and the
        caller reports them.
        """
        max_pages = max_pages or config.SCRAPER_MAX_LISTING_PAGES
        wanted = {_normalize_name(t): t for t in targets}
        resolved: dict[str, str] = {}

        for page in range(1, max_pages + 1):
            if len(resolved) == len(wanted):
                break

            url = self._listing_page_url(page)
            logger.info("Discovery page %s/%s: %s", page, max_pages, url)
            soup = self.fetch(url)
            if soup is None:
                continue

            entries = self.parse_listing_page(soup)
            if not entries:
                logger.warning("No phone entries on %s; stopping discovery", url)
                break

            for listing_name, phone_url in entries.items():
                key = _normalize_name(listing_name)
                if key in wanted and wanted[key] not in resolved:
                    resolved[wanted[key]] = phone_url
                    logger.info("  matched %s -> %s", wanted[key], phone_url)

        missing = sorted(set(wanted.values()) - set(resolved))
        if missing:
            logger.warning("Not found in listing: %s", ", ".join(missing))

        return resolved

    # ------------------------------------------------------------------
    # Stage 2: extraction
    # ------------------------------------------------------------------
    def parse_spec_page(self, soup: BeautifulSoup, url: str) -> ScrapedPhone | None:
        """Parse a phone detail page into a `ScrapedPhone`."""
        title_el = soup.select_one("h1.specs-phone-name-title")
        if not title_el:
            logger.warning("No title on %s; page layout may have changed", url)
            return None
        name = clean_text(title_el.get_text(strip=True))
        if not name:
            return None

        # `specs[category][label] = value`, mirroring GSMArena's own grouping.
        specs: dict[str, dict[str, str]] = {}
        raw_specs: list[dict] = []

        for table in soup.select("#specs-list table"):
            header = table.find("th")
            category = clean_text(header.get_text(strip=True)) if header else "Other"
            category = category or "Other"
            bucket = specs.setdefault(category, {})

            # GSMArena leaves the label cell empty for continuation rows (extra
            # SIM options, secondary camera lines). Those are kept under a
            # numbered key so no published value is dropped.
            unlabeled = 0
            for row in table.select("tr"):
                label_cell = row.select_one("td.ttl")
                value_cell = row.select_one("td.nfo")
                if value_cell is None:
                    continue

                value = clean_text(value_cell.get_text(" ", strip=True))
                if not value:
                    continue

                label = clean_text(label_cell.get_text(" ", strip=True)) if label_cell else None
                if not label:
                    unlabeled += 1
                    label = f"Additional {unlabeled}"

                bucket.setdefault(label, value)
                raw_specs.append(
                    {"category": category, "name": label, "value": value}
                )

        if not raw_specs:
            logger.warning("No specs parsed from %s", url)
            return None

        def spec(category: str, label: str) -> str | None:
            return specs.get(category, {}).get(label)

        # The camera section header doubles as the lens count ("Triple", "Quad"),
        # and the value under it is the lens list itself.
        main_camera_setup, main_camera = self._camera_block(specs, "Main Camera")
        _, selfie_camera = self._camera_block(specs, "Selfie camera")

        internal_memory = spec("Memory", "Internal")
        max_ram, max_storage = parse_memory_options(internal_memory)

        display_type = spec("Display", "Type")
        battery_type = spec("Battery", "Type")
        charging = spec("Battery", "Charging")
        announced = spec("Launch", "Announced")

        price_text = spec("Misc", "Price") or self._fallback_price(soup)

        image_el = soup.select_one(".specs-photo-main img")

        phone = {
            "name": name,
            "slug": url.rstrip("/").split("/")[-1].replace(".php", ""),
            "brand": "Samsung",
            "url": url,
            "image_url": image_el.get("src") if image_el else None,
            "announced": announced,
            "status": spec("Launch", "Status"),
            "release_year": parse_release_year(announced),
            "dimensions": spec("Body", "Dimensions"),
            "weight": spec("Body", "Weight"),
            "weight_g": parse_weight_grams(spec("Body", "Weight")),
            "build": spec("Body", "Build"),
            "sim": spec("Body", "SIM"),
            "display_type": display_type,
            "display_size": spec("Display", "Size"),
            "display_size_inches": parse_display_inches(spec("Display", "Size")),
            "display_resolution": spec("Display", "Resolution"),
            "display_refresh_rate_hz": parse_refresh_rate(display_type),
            "display_protection": spec("Display", "Protection"),
            "os": spec("Platform", "OS"),
            "chipset": spec("Platform", "Chipset"),
            "cpu": spec("Platform", "CPU"),
            "gpu": spec("Platform", "GPU"),
            "card_slot": spec("Memory", "Card slot"),
            "internal_memory": internal_memory,
            "max_ram_gb": max_ram,
            "max_storage_gb": max_storage,
            "main_camera_setup": main_camera_setup,
            "main_camera": main_camera,
            "main_camera_mp": parse_camera_mp(main_camera),
            "main_camera_features": spec("Main Camera", "Features"),
            "main_camera_video": spec("Main Camera", "Video"),
            "selfie_camera": selfie_camera,
            "selfie_camera_mp": parse_camera_mp(selfie_camera),
            "selfie_camera_video": spec("Selfie camera", "Video"),
            "loudspeaker": spec("Sound", "Loudspeaker"),
            "jack_3_5mm": spec("Sound", "3.5mm jack"),
            "wlan": spec("Comms", "WLAN"),
            "bluetooth": spec("Comms", "Bluetooth"),
            "nfc": spec("Comms", "NFC"),
            "usb": spec("Comms", "USB"),
            "sensors": spec("Features", "Sensors"),
            "battery_type": battery_type,
            "battery_capacity_mah": parse_battery_mah(battery_type),
            "charging": charging,
            "charging_watts": parse_charging_watts(charging),
            "colors": spec("Misc", "Colors"),
            "models": spec("Misc", "Models"),
            "price_text": price_text,
        }

        return ScrapedPhone(
            phone=phone, raw_specs=raw_specs, prices=parse_prices(price_text)
        )

    @staticmethod
    def _camera_block(
        specs: dict[str, dict[str, str]], category: str
    ) -> tuple[str | None, str | None]:
        """Return `(setup, lens list)` for a camera section.

        GSMArena labels the lens row by configuration — "Single", "Dual",
        "Triple", "Quad" — so the label is the setup and the value is the list.
        """
        block = specs.get(category, {})
        for setup in ("Quad", "Triple", "Dual", "Single"):
            if setup in block:
                return setup, block[setup]
        # Fall back to the first non-metadata row if Samsung uses another label.
        for label, value in block.items():
            if label not in ("Features", "Video"):
                return label, value
        return None, None

    @staticmethod
    def _fallback_price(soup: BeautifulSoup) -> str | None:
        """Read the price widget when the Misc/Price cell is blank.

        The Misc row is often an empty placeholder while the actual figures sit
        in the pricing box further down the page.
        """
        prices = [
            clean_text(el.get_text(" ", strip=True))
            for el in soup.select(".pricing-item-price, #pricing-info .price")
        ]
        values = [p for p in prices if p]
        return " / ".join(values[:4]) if values else None

    def scrape_phone(self, url: str) -> ScrapedPhone | None:
        soup = self.fetch(url)
        if soup is None:
            return None
        return self.parse_spec_page(soup, url)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def scrape(self, targets: Iterable[str] | None = None) -> list[ScrapedPhone]:
        """Discover and scrape every target model. Returns what succeeded."""
        targets = list(targets or config.TARGET_MODELS)
        logger.info("Scraping %s Samsung models from GSMArena", len(targets))

        urls = self.discover_phone_urls(targets)
        results: list[ScrapedPhone] = []

        for index, (model, url) in enumerate(urls.items(), start=1):
            logger.info("[%s/%s] Scraping %s", index, len(urls), model)
            scraped = self.scrape_phone(url)
            if scraped is None:
                logger.error("Failed to parse %s (%s)", model, url)
                continue
            logger.info(
                "  %s: %s specs, battery=%s mAh, display=%s in",
                scraped.name,
                len(scraped.raw_specs),
                scraped.phone.get("battery_capacity_mah"),
                scraped.phone.get("display_size_inches"),
            )
            results.append(scraped)

        logger.info("Scraped %s/%s models", len(results), len(targets))
        return results

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "GSMArenaScraper":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
