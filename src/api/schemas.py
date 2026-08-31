"""Pydantic request and response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="A question about Samsung phones",
        examples=["What are the camera specs of the Samsung Galaxy S23?"],
    )
    top_k: int | None = Field(
        None, ge=1, le=20, description="How many passages to retrieve"
    )


class ReviewRequest(BaseModel):
    phone: str = Field(
        ..., min_length=1, description="Model name", examples=["Galaxy S23 Ultra"]
    )
    audience: str = Field(
        "a general buyer",
        description="Who the review is written for",
        examples=["a mobile photographer"],
    )


class CompareRequest(BaseModel):
    phone_a: str = Field(..., min_length=1, examples=["Galaxy S23"])
    phone_b: str = Field(..., min_length=1, examples=["Galaxy S22"])
    focus: str = Field(
        "overall",
        description="Aspect to focus on: performance, camera, battery, display, price",
        examples=["performance"],
    )


class AgentQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Free-text request routed to the right agent crew",
        examples=["Review the Galaxy S24 Ultra for a photographer"],
    )


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------
class SourceRef(BaseModel):
    phone: str
    aspect: str
    score: float | None = None


class ChatResponseModel(BaseModel):
    answer: str
    intent: str
    phones: list[str] = []
    aspects: list[str] = []
    sources: list[SourceRef] = []
    generated_by: str


class PriceModel(BaseModel):
    currency: str
    amount: float
    raw_text: str | None = None
    source: str | None = None


class PhoneSummary(BaseModel):
    id: int
    name: str
    release_year: int | None = None
    display_size_inches: float | None = None
    chipset: str | None = None
    main_camera_mp: float | None = None
    battery_capacity_mah: int | None = None
    url: str | None = None


class PhoneDetail(PhoneSummary):
    announced: str | None = None
    status: str | None = None
    dimensions: str | None = None
    weight: str | None = None
    build: str | None = None
    display_type: str | None = None
    display_size: str | None = None
    display_resolution: str | None = None
    display_refresh_rate_hz: int | None = None
    display_protection: str | None = None
    os: str | None = None
    cpu: str | None = None
    gpu: str | None = None
    internal_memory: str | None = None
    max_ram_gb: int | None = None
    max_storage_gb: int | None = None
    main_camera: str | None = None
    main_camera_video: str | None = None
    selfie_camera: str | None = None
    selfie_camera_mp: float | None = None
    battery_type: str | None = None
    charging: str | None = None
    charging_watts: float | None = None
    colors: str | None = None
    prices: list[PriceModel] = []
    specifications: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Complete raw GSMArena specs, grouped by category",
    )


class ReviewResponse(BaseModel):
    success: bool
    phone: str | None = None
    review: str
    specifications: str | None = None
    key_numbers: dict[str, Any] = {}
    generated_by: str | None = None
    framework: str | None = Field(
        None, description="Which orchestrator ran the crew: crewai or native"
    )
    agents: dict[str, Any] = {}


class CompareResponse(BaseModel):
    success: bool
    phones: list[str] = []
    focus: str
    table: str | None = None
    comparison: str
    generated_by: str | None = None
    framework: str | None = None
    agents: dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    database: dict[str, Any]
    vector_store: dict[str, Any]
    llm: dict[str, Any]
    agents: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    detail: str
