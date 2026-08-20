"""Pydantic sxemalar — javob shakli va avtomatik OpenAPI hujjati uchun."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    backend: str = Field(..., examples=["onnx"])
    model: str | None = Field(None, examples=["resnet18"])
    classes: int = Field(..., examples=[16])
    min_confidence: float = Field(..., examples=[0.45])


class ClassesResponse(BaseModel):
    classes: list[str]
    backend: str
    min_confidence: float


class Candidate(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    filename: str | None
    label: str = Field(..., examples=["osh"], description="Eng yuqori ballli sinf")
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.9731])

    # Softmax har doim yig'indisi 1 ga teng taqsimot qaytaradi — model
    # ro'yxatda yo'q taomni ham majburan mavjud sinflardan biriga tiqadi.
    # Shuning uchun javobda "ishonchlimi" degan alohida bayroq bor.
    is_confident: bool = Field(
        ..., description="confidence >= min_confidence bo'lsa true"
    )
    message: str | None = Field(
        None, description="Ishonch past bo'lganda foydalanuvchiga izoh"
    )
    top_k: list[Candidate] = Field(..., description="Eng yuqori ballli sinflar")

    scores: dict[str, float]
    latency_ms: float = Field(..., description="Faqat model inference vaqti")
    backend: str


class BatchResponse(BaseModel):
    count: int
    total_latency_ms: float
    results: list[PredictResponse]
