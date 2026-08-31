"""SQLAlchemy schema for scraped Samsung phone data.

The design deliberately keeps two views of the same scrape:

* `phones` flattens the specs an analyst actually filters and sorts on
  (battery capacity, display size, chipset), with numeric columns parsed out of
  GSMArena's free text so queries like "best battery life" are plain SQL.
* `specifications` preserves every key/value GSMArena publishes, unmodified, so
  nothing is lost to the flattening and the RAG layer has full detail to
  retrieve over.

`prices` is separate because one phone carries several regional listings.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Phone(Base):
    """One Samsung handset, with the headline specs flattened into columns."""

    __tablename__ = "phones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False, unique=True, index=True)
    slug = Column(String(180), nullable=False, unique=True)
    brand = Column(String(60), nullable=False, default="Samsung")
    url = Column(String(400), nullable=False)
    image_url = Column(String(400))

    # Launch
    announced = Column(String(120))
    status = Column(String(160))
    release_year = Column(Integer, index=True)

    # Body
    dimensions = Column(String(200))
    weight = Column(String(80))
    weight_g = Column(Float)
    build = Column(Text)
    sim = Column(Text)

    # Display
    display_type = Column(Text)
    display_size = Column(String(200))
    display_size_inches = Column(Float, index=True)
    display_resolution = Column(String(200))
    display_refresh_rate_hz = Column(Integer)
    display_protection = Column(String(200))

    # Platform
    os = Column(String(250))
    chipset = Column(String(250), index=True)
    cpu = Column(Text)
    gpu = Column(String(200))

    # Memory
    card_slot = Column(String(120))
    internal_memory = Column(Text)
    max_ram_gb = Column(Integer)
    max_storage_gb = Column(Integer)

    # Cameras
    main_camera_setup = Column(String(80))
    main_camera = Column(Text)
    main_camera_mp = Column(Float)
    main_camera_features = Column(Text)
    main_camera_video = Column(Text)
    selfie_camera = Column(Text)
    selfie_camera_mp = Column(Float)
    selfie_camera_video = Column(Text)

    # Sound & comms
    loudspeaker = Column(String(200))
    jack_3_5mm = Column(String(80))
    wlan = Column(Text)
    bluetooth = Column(String(200))
    nfc = Column(String(80))
    usb = Column(String(200))

    # Features
    sensors = Column(Text)

    # Battery
    battery_type = Column(String(200))
    battery_capacity_mah = Column(Integer, index=True)
    charging = Column(Text)
    charging_watts = Column(Float)

    # Misc
    colors = Column(Text)
    models = Column(Text)
    price_text = Column(String(200))

    scraped_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    specifications = relationship(
        "Specification",
        back_populates="phone",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    prices = relationship(
        "Price",
        back_populates="phone",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Phone {self.name!r} battery={self.battery_capacity_mah}mAh>"

    def to_dict(self, include_raw_specs: bool = False) -> dict:
        data = {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
        data["prices"] = [p.to_dict() for p in self.prices]
        if include_raw_specs:
            data["raw_specifications"] = [s.to_dict() for s in self.specifications]
        return data

    def spec_summary(self) -> str:
        """Compact human-readable spec block used in prompts and reviews."""
        rows = [
            ("Display", self.display_size),
            ("Display type", self.display_type),
            ("Resolution", self.display_resolution),
            ("Chipset", self.chipset),
            ("CPU", self.cpu),
            ("GPU", self.gpu),
            ("Memory", self.internal_memory),
            ("Main camera", self.main_camera),
            ("Selfie camera", self.selfie_camera),
            ("Video", self.main_camera_video),
            ("Battery", f"{self.battery_type or ''} {self.charging or ''}".strip()),
            ("OS", self.os),
            ("Build", self.build),
            ("Weight", self.weight),
            ("Announced", self.announced),
        ]
        lines = [f"- {label}: {value}" for label, value in rows if value]
        return f"{self.name}\n" + "\n".join(lines)


class Specification(Base):
    """Every raw `category / name / value` triple GSMArena lists for a phone."""

    __tablename__ = "specifications"
    __table_args__ = (
        Index("ix_spec_phone_category", "phone_id", "category"),
        UniqueConstraint("phone_id", "category", "name", name="uq_spec_identity"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_id = Column(
        Integer, ForeignKey("phones.id", ondelete="CASCADE"), nullable=False
    )
    category = Column(String(80), nullable=False)
    name = Column(String(120), nullable=False)
    value = Column(Text, nullable=False)

    phone = relationship("Phone", back_populates="specifications")

    def to_dict(self) -> dict:
        return {"category": self.category, "name": self.name, "value": self.value}

    def __repr__(self) -> str:
        return f"<Spec {self.category}/{self.name}>"


class Price(Base):
    """A regional price listing. GSMArena quotes several per phone."""

    __tablename__ = "prices"
    __table_args__ = (Index("ix_price_phone", "phone_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_id = Column(
        Integer, ForeignKey("phones.id", ondelete="CASCADE"), nullable=False
    )
    currency = Column(String(10), nullable=False)
    amount = Column(Float, nullable=False)
    raw_text = Column(String(200))
    source = Column(String(80), default="GSMArena")

    phone = relationship("Phone", back_populates="prices")

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "amount": self.amount,
            "raw_text": self.raw_text,
            "source": self.source,
        }

    def __repr__(self) -> str:
        return f"<Price {self.currency}{self.amount}>"
