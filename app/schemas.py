"""Pydantic sxemalar — javob shakli va avtomatik OpenAPI hujjati uchun."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    backend: str = Field(..., examples=["onnx"])
    model: str | None = Field(None, examples=["resnet18"])
    classes: int = Field(..., examples=[5])


class ClassesResponse(BaseModel):
    classes: list[str]
    backend: str


class PredictResponse(BaseModel):
    filename: str | None
    label: str = Field(..., examples=["osh"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.9731])
    scores: dict[str, float]
    latency_ms: float = Field(..., description="Faqat model inference vaqti")
    backend: str


class BatchResponse(BaseModel):
    count: int
    total_latency_ms: float
    results: list[PredictResponse]
