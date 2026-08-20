"""
PyTorch checkpoint -> ONNX.

    python scripts/export_onnx.py --checkpoint ../uzbek-food-classifier/reports/resnet18/best.pt

Nima uchun ONNX:
  * Docker obrazi ~800 MB o'rniga ~250 MB (PyTorch kerak emas)
  * CPU'da inference tezroq — ONNX Runtime grafni optimallashtiradi
    (Conv+BatchNorm birlashtiriladi, keraksiz tugunlar tashlanadi)
  * Sovuq start ancha tez: torch import qilishning o'ziga ~2 sekund ketadi

Eksportdan keyin `--check` bilan ikkala model bir xil natija berishini
tekshiramiz. Bu bosqichni o'tkazib yuborish mumkin emas: opset mos
kelmasligi yoki dinamik shakl xatosi jimgina noto'g'ri natija beradi.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.torch_model import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX eksport")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    arch = checkpoint["arch"]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Arxitektura: {arch}, sinflar: {classes}")

    model = build_model(arch, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    onnx_path = args.out_dir / "model.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        # Dinamik batch: xizmat bir vaqtda bir nechta rasmni qayta ishlashi
        # mumkin bo'lishi kerak, aks holda batch=1 ga qotib qoladi.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"ONNX saqlandi: {onnx_path} ({size_mb:.1f} MB)")

    # Checkpoint va yorliqlarni ham xizmat papkasiga ko'chiramiz
    meta = {
        "classes": classes,
        "arch": arch,
        "image_size": args.image_size,
        "opset": args.opset,
        "source_checkpoint": str(args.checkpoint),
        "val_acc": checkpoint.get("val_acc"),
    }
    (args.out_dir / "labels.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    target_ckpt = args.out_dir / "best.pt"
    if args.checkpoint.resolve() != target_ckpt.resolve():
        shutil.copy2(args.checkpoint, target_ckpt)
    print(f"Meta: {args.out_dir / 'labels.json'}")

    if args.no_check:
        return

    # --- Ekvivalentlikni tekshirish ---
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_diff = 0.0
    for _ in range(5):
        sample = torch.randn(2, 3, args.image_size, args.image_size)
        with torch.no_grad():
            torch_out = model(sample).numpy()
        onnx_out = session.run(None, {input_name: sample.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(torch_out - onnx_out).max()))

    print(f"PyTorch va ONNX orasidagi maksimal farq: {max_diff:.2e}")
    # 1e-4 — fp32 hisob-kitobdagi tabiiy tafovut chegarasi. Bundan katta
    # farq eksport xatosidan darak beradi.
    if max_diff > 1e-4:
        print("OGOHLANTIRISH: farq kutilganidan katta, eksportni tekshiring!")
        sys.exit(1)
    print("Tekshiruv o'tdi: ikkala model bir xil natija beradi.")


if __name__ == "__main__":
    main()
