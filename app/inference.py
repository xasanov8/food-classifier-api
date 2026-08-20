"""
Inference: ONNX Runtime (asosiy) va PyTorch (ixtiyoriy) — bitta interfeys ortida.

Production'da ONNX ishlatiladi: obraz kichik, sovuq start tez, CPU'da
inference sezilarli tezroq. PyTorch backend faqat solishtirish va
export'ni tekshirish uchun kerak, shuning uchun u ixtiyoriy import.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.preprocessing import preprocess_bytes, softmax

# Model yo'li joriy ish katalogiga EMAS, paketning o'ziga nisbatan
# aniqlanadi. Aks holda xizmat qayerdan ishga tushirilganiga bog'liq
# bo'lib qoladi: `uvicorn` ni boshqa katalogdan chaqirsangiz yoki
# supervisor/systemd boshqa cwd bilan ishga tushirsa, model topilmaydi.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR") or _PACKAGE_ROOT / "models")


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]
    latency_ms: float


class Classifier:
    """Backend'dan qat'i nazar bir xil interfeys."""

    def __init__(self, backend: str = "onnx", models_dir: Path = MODELS_DIR) -> None:
        self.backend = backend
        self.models_dir = Path(models_dir)

        meta_file = self.models_dir / "labels.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"{meta_file} topilmadi. Avval modelni eksport qiling: "
                "python scripts/export_onnx.py"
            )
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes"]
        self.arch: str = meta.get("arch", "resnet18")

        if backend == "onnx":
            self._init_onnx()
        elif backend == "torch":
            self._init_torch()
        else:
            raise ValueError(f"noma'lum backend: {backend}")

    # ---------- ONNX ----------
    def _init_onnx(self) -> None:
        import onnxruntime as ort

        path = self.models_dir / "model.onnx"
        if not path.exists():
            raise FileNotFoundError(f"{path} topilmadi — scripts/export_onnx.py ishlating")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Konteynerda odatda 1-2 vCPU bo'ladi; ko'p oqim faqat zarar qiladi.
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))

        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _run_onnx(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch})[0]

    # ---------- PyTorch ----------
    def _init_torch(self) -> None:
        import torch

        from app.torch_model import build_model

        path = self.models_dir / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.model = build_model(checkpoint["arch"], len(self.classes))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        # Inference'da grad kerak emas — global o'chirib qo'yamiz.
        self.torch.set_grad_enabled(False)

    def _run_torch(self, batch: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(batch)
        return self.model(tensor).numpy()

    # ---------- Umumiy ----------
    def predict_array(self, batch: np.ndarray) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        logits = self._run_onnx(batch) if self.backend == "onnx" else self._run_torch(batch)
        return logits, (time.perf_counter() - started) * 1000.0

    def predict(self, raw: bytes) -> Prediction:
        batch = preprocess_bytes(raw)
        logits, latency_ms = self.predict_array(batch)
        probs = softmax(logits)[0]
        index = int(probs.argmax())
        return Prediction(
            label=self.classes[index],
            confidence=float(probs[index]),
            scores={name: float(p) for name, p in zip(self.classes, probs)},
            latency_ms=round(latency_ms, 2),
        )

    def warmup(self, runs: int = 3) -> None:
        """Birinchi so'rov har doim sekin (lazy init, xotira ajratish).
        Startupda bir necha marta bo'sh o'tkazamiz."""
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        for _ in range(runs):
            self.predict_array(dummy)
