"""
FastAPI xizmati — o'zbek milliy taomlari klassifikatori.

    uvicorn app.main:app --reload

Endpointlar:
    GET  /            -> rasm yuklash sahifasi
    GET  /health      -> tiriklik tekshiruvi (Docker healthcheck shuni chaqiradi)
    GET  /classes     -> model biladigan sinflar
    POST /predict     -> bitta rasm
    POST /predict/batch -> bir nechta rasm
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.inference import Classifier
from app.schemas import (
    BatchResponse,
    Candidate,
    ClassesResponse,
    HealthResponse,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("food-api")

BACKEND = os.getenv("BACKEND", "onnx")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))  # 8 MB
MAX_BATCH = int(os.getenv("MAX_BATCH", "16"))
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Model bitta marta yuklanadi va state'da saqlanadi. Har so'rovda qayta
# yuklash ~300 ms qo'shadi va xotirani behuda sarflaydi.
state: dict[str, Classifier] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Model yuklanmoqda (backend=%s)...", BACKEND)
    classifier = Classifier(backend=BACKEND)
    classifier.warmup()
    state["classifier"] = classifier
    logger.info("Tayyor. %d ta sinf, ishonch chegarasi %.2f",
                len(classifier.classes), classifier.min_confidence)
    yield
    state.clear()


app = FastAPI(
    title="O'zbek taomlari klassifikatori",
    description=(
        "O'zbek milliy taomlarini rasmdan aniqlaydi. ResNet fine-tuned, "
        "ONNX Runtime orqali xizmat qiladi. Sinflar ro'yxati: GET /classes. "
        "Ishonch chegarasidan past natijalar is_confident=false bilan belgilanadi."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


LOW_CONFIDENCE_MESSAGE = (
    "Ishonch past. Bu taom modelga tanish bo'lmasligi yoki rasm noaniq "
    "bo'lishi mumkin. Quyida eng yaqin variantlar keltirildi."
)


def build_response(
    classifier: Classifier, filename: str | None, result
) -> PredictResponse:
    return PredictResponse(
        filename=filename,
        label=result.label,
        confidence=round(result.confidence, 4),
        is_confident=result.is_confident,
        message=None if result.is_confident else LOW_CONFIDENCE_MESSAGE,
        top_k=[Candidate(label=name, score=round(score, 4))
               for name, score in result.top_k],
        scores={k: round(v, 4) for k, v in result.scores.items()},
        latency_ms=result.latency_ms,
        backend=classifier.backend,
    )


def get_classifier() -> Classifier:
    classifier = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model hali yuklanmagan")
    return classifier


async def read_image(file: UploadFile) -> bytes:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Rasm kutilgan edi, keldi: {file.content_type}",
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Bo'sh fayl")
    # Hajm chegarasi: cheklovsiz upload xizmatni xotira bilan yiqitishi mumkin.
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fayl juda katta ({len(raw)} bayt, chegara {MAX_UPLOAD_BYTES})",
        )
    return raw


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="static/index.html topilmadi")
    return FileResponse(page)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    classifier = state.get("classifier")
    return HealthResponse(
        status="ok" if classifier else "loading",
        backend=BACKEND,
        model=classifier.arch if classifier else None,
        classes=len(classifier.classes) if classifier else 0,
        min_confidence=classifier.min_confidence if classifier else 0.0,
    )


@app.get("/classes", response_model=ClassesResponse)
async def classes() -> ClassesResponse:
    classifier = get_classifier()
    return ClassesResponse(
        classes=classifier.classes,
        backend=classifier.backend,
        min_confidence=classifier.min_confidence,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    classifier = get_classifier()
    raw = await read_image(file)
    try:
        result = classifier.predict(raw)
    except Exception as exc:
        logger.exception("Bashorat xatosi")
        raise HTTPException(status_code=400, detail=f"Rasmni o'qib bo'lmadi: {exc}") from exc

    return build_response(classifier, file.filename, result)


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(files: list[UploadFile] = File(...)) -> BatchResponse:
    classifier = get_classifier()
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=413, detail=f"Ko'pi bilan {MAX_BATCH} ta rasm"
        )

    results = []
    total_ms = 0.0
    for file in files:
        raw = await read_image(file)
        try:
            result = classifier.predict(raw)
        except Exception as exc:
            logger.warning("O'tkazib yuborildi %s: %s", file.filename, exc)
            continue
        total_ms += result.latency_ms
        results.append(build_response(classifier, file.filename, result))

    if not results:
        raise HTTPException(status_code=400, detail="Hech qaysi rasm o'qilmadi")

    return BatchResponse(
        count=len(results),
        total_latency_ms=round(total_ms, 2),
        results=results,
    )
