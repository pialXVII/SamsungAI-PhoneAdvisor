"""Build the retrievable document set from database rows.

Each phone becomes several *aspect* documents (display, camera, performance,
battery, …) rather than one blob. A question about cameras should retrieve the
camera paragraph of the right phone, not a 60-line spec dump where the relevant
sentence is diluted by everything else — one embedding per aspect keeps the
vector focused on a single topic.

Every document carries its `phone_id`, so an answer can always be traced back to
the exact database row it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from src.database.models import Phone
from src.database.repository import get_all_phones


@dataclass
class Document:
    """One retrievable passage plus the metadata used for citation."""

    text: str
    phone_id: int
    phone_name: str
    aspect: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Document {self.phone_name}/{self.aspect}>"


def _joined(*parts: str | None) -> str:
    return " ".join(p for p in parts if p)


def _price_line(phone: Phone) -> str | None:
    if not phone.prices:
        return phone.price_text
    listed = ", ".join(f"{p.currency} {p.amount:,.2f}" for p in phone.prices)
    return f"Approximate market price: {listed}."


def build_documents_for_phone(phone: Phone) -> list[Document]:
    """Produce the per-aspect passages for one handset."""
    name = phone.name
    documents: list[Document] = []

    def add(aspect: str, body: str | None, **metadata) -> None:
        if not body or not body.strip():
            return
        # Leading with the model name in every passage keeps the phone
        # identifiable after retrieval detaches the chunk from its context.
        documents.append(
            Document(
                text=f"{name} — {aspect}. {body.strip()}",
                phone_id=phone.id,
                phone_name=name,
                aspect=aspect,
                metadata=metadata,
            )
        )

    add(
        "Overview",
        _joined(
            f"{name} is a Samsung smartphone announced {phone.announced}."
            if phone.announced
            else f"{name} is a Samsung smartphone.",
            f"Status: {phone.status}." if phone.status else None,
            f"It has a {phone.display_size} display" if phone.display_size else None,
            f"powered by {phone.chipset}" if phone.chipset else None,
            f"with a {phone.battery_type} battery." if phone.battery_type else None,
            _price_line(phone),
        ),
        release_year=phone.release_year,
    )

    add(
        "Display",
        _joined(
            f"Screen type: {phone.display_type}." if phone.display_type else None,
            f"Size: {phone.display_size}." if phone.display_size else None,
            f"Resolution: {phone.display_resolution}."
            if phone.display_resolution
            else None,
            f"Refresh rate: {phone.display_refresh_rate_hz} Hz."
            if phone.display_refresh_rate_hz
            else None,
            f"Protection: {phone.display_protection}."
            if phone.display_protection
            else None,
        ),
        display_size_inches=phone.display_size_inches,
    )

    add(
        "Camera",
        _joined(
            f"Main camera ({phone.main_camera_setup}): {phone.main_camera}."
            if phone.main_camera
            else None,
            f"Camera features: {phone.main_camera_features}."
            if phone.main_camera_features
            else None,
            f"Main camera video recording: {phone.main_camera_video}."
            if phone.main_camera_video
            else None,
            f"Selfie camera: {phone.selfie_camera}." if phone.selfie_camera else None,
            f"Selfie video: {phone.selfie_camera_video}."
            if phone.selfie_camera_video
            else None,
        ),
        main_camera_mp=phone.main_camera_mp,
    )

    add(
        "Performance",
        _joined(
            f"Chipset: {phone.chipset}." if phone.chipset else None,
            f"CPU: {phone.cpu}." if phone.cpu else None,
            f"GPU: {phone.gpu}." if phone.gpu else None,
            f"Memory configurations: {phone.internal_memory}."
            if phone.internal_memory
            else None,
            f"Operating system: {phone.os}." if phone.os else None,
            f"Expandable storage: {phone.card_slot}." if phone.card_slot else None,
        ),
        chipset=phone.chipset,
        max_ram_gb=phone.max_ram_gb,
    )

    add(
        "Battery and charging",
        _joined(
            f"Battery: {phone.battery_type}." if phone.battery_type else None,
            f"Capacity: {phone.battery_capacity_mah} mAh."
            if phone.battery_capacity_mah
            else None,
            f"Charging: {phone.charging}." if phone.charging else None,
        ),
        battery_capacity_mah=phone.battery_capacity_mah,
    )

    add(
        "Design and build",
        _joined(
            f"Dimensions: {phone.dimensions}." if phone.dimensions else None,
            f"Weight: {phone.weight}." if phone.weight else None,
            f"Build: {phone.build}." if phone.build else None,
            f"SIM: {phone.sim}." if phone.sim else None,
            f"Available colors: {phone.colors}." if phone.colors else None,
        ),
        weight_g=phone.weight_g,
    )

    add(
        "Connectivity and features",
        _joined(
            f"Wi-Fi: {phone.wlan}." if phone.wlan else None,
            f"Bluetooth: {phone.bluetooth}." if phone.bluetooth else None,
            f"NFC: {phone.nfc}." if phone.nfc else None,
            f"USB: {phone.usb}." if phone.usb else None,
            f"3.5mm headphone jack: {phone.jack_3_5mm}." if phone.jack_3_5mm else None,
            f"Loudspeaker: {phone.loudspeaker}." if phone.loudspeaker else None,
            f"Sensors: {phone.sensors}." if phone.sensors else None,
        ),
    )

    add("Pricing", _price_line(phone))

    return documents


def build_corpus(session: Session) -> list[Document]:
    """Build the full document set for every phone in the database."""
    documents: list[Document] = []
    for phone in get_all_phones(session):
        documents.extend(build_documents_for_phone(phone))
    return documents
