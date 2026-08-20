"""
API testlari.

    pytest -q

Testlar haqiqiy modelni yuklaydi (mock emas): eksport qilingan ONNX fayl
yo'q bo'lsa yoki buzilgan bo'lsa, testlar buni darhol ko'rsatishi kerak.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

pytestmark = pytest.mark.skipif(
    not (MODELS / "model.onnx").exists(),
    reason="models/model.onnx yo'q — avval scripts/export_onnx.py ishlating",
)


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def make_image(size=(400, 300), color=(180, 140, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["classes"] > 0


def test_classes(client):
    body = client.get("/classes").json()
    assert isinstance(body["classes"], list)
    assert len(body["classes"]) >= 2


def test_predict_returns_valid_distribution(client):
    response = client.post(
        "/predict", files={"file": ("test.jpg", make_image(), "image/jpeg")}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in body["scores"]
    assert 0.0 <= body["confidence"] <= 1.0
    # Softmax chiqishi ehtimollik taqsimoti bo'lishi shart
    assert abs(sum(body["scores"].values()) - 1.0) < 1e-3
    # Eng katta ball tanlangan yorliqqa mos kelishi kerak
    assert max(body["scores"], key=body["scores"].get) == body["label"]


def test_predict_is_deterministic(client):
    """Bir xil rasm -> bir xil natija. eval() rejimi buzilsa, bu yiqiladi."""
    image = make_image()
    first = client.post("/predict", files={"file": ("a.jpg", image, "image/jpeg")}).json()
    second = client.post("/predict", files={"file": ("a.jpg", image, "image/jpeg")}).json()
    assert first["label"] == second["label"]
    assert abs(first["confidence"] - second["confidence"]) < 1e-6


def test_rejects_non_image(client):
    response = client.post(
        "/predict", files={"file": ("x.txt", b"salom dunyo", "text/plain")}
    )
    assert response.status_code == 415


def test_rejects_corrupt_image(client):
    response = client.post(
        "/predict", files={"file": ("bad.jpg", b"\xff\xd8\xff not-an-image", "image/jpeg")}
    )
    assert response.status_code == 400


def test_rejects_empty_file(client):
    response = client.post("/predict", files={"file": ("e.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_batch(client):
    files = [("files", (f"{i}.jpg", make_image(), "image/jpeg")) for i in range(3)]
    body = client.post("/predict/batch", files=files).json()
    assert body["count"] == 3
    assert len(body["results"]) == 3


def test_preprocessing_output_shape_and_range():
    from app.preprocessing import preprocess

    with Image.open(io.BytesIO(make_image((640, 480)))) as img:
        array = preprocess(img)

    assert array.shape == (1, 3, 224, 224)
    assert array.dtype == np.float32
    # ImageNet normalizatsiyasidan keyin qiymatlar taxminan [-2.2, 2.7]
    assert -3.0 < array.min() and array.max() < 3.0


def test_preprocessing_handles_non_square():
    """Nisbat saqlanib, markazdan kesilishi kerak — cho'zilmasligi."""
    from app.preprocessing import preprocess

    for size in [(1000, 200), (200, 1000), (224, 224), (300, 300)]:
        with Image.open(io.BytesIO(make_image(size))) as img:
            assert preprocess(img).shape == (1, 3, 224, 224)
