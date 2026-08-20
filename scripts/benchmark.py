"""
PyTorch vs ONNX Runtime — inference tezligini o'lchash.

    python scripts/benchmark.py --runs 200

Adolatli o'lchov uchun uchta shart:
  1. Warmup — birinchi bir necha chaqiruv har doim sekin (lazy init,
     xotira ajratish, kesh isishi). Ular o'lchovga kirmaydi.
  2. Median va p95 — o'rtacha arifmetik bitta tasodifiy sakrashdan
     buziladi. Latency har doim taqsimot bilan hisobot qilinadi.
  3. Bir xil kirish — ikkala backend aynan bir xil massivni ko'radi.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference import Classifier  # noqa: E402


def bench(classifier: Classifier, batch: np.ndarray, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        classifier.predict_array(batch)

    timings: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        classifier.predict_array(batch)
        timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    return {
        "runs": runs,
        "mean_ms": round(statistics.mean(timings), 3),
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(timings[int(len(timings) * 0.95) - 1], 3),
        "min_ms": round(timings[0], 3),
        "max_ms": round(timings[-1], 3),
        "throughput_img_s": round(batch.shape[0] * 1000.0 / statistics.median(timings), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    results: dict = {"backends": {}}

    for backend in ("torch", "onnx"):
        try:
            classifier = Classifier(backend=backend, models_dir=args.models_dir)
        except Exception as exc:
            print(f"{backend}: o'tkazib yuborildi ({exc})")
            continue

        results["backends"][backend] = {}
        for batch_size in args.batch_sizes:
            batch = rng.standard_normal((batch_size, 3, 224, 224)).astype(np.float32)
            stats = bench(classifier, batch, args.runs, args.warmup)
            results["backends"][backend][f"batch{batch_size}"] = stats
            print(
                f"{backend:<6} batch={batch_size:<2}  "
                f"median={stats['median_ms']:>7.2f} ms  "
                f"p95={stats['p95_ms']:>7.2f} ms  "
                f"{stats['throughput_img_s']:>7.1f} rasm/s"
            )

    # --- Xulosa jadvali ---
    lines = ["| Backend | Batch | Median (ms) | p95 (ms) | Rasm/s |", "|---|---|---|---|---|"]
    for backend, batches in results["backends"].items():
        for name, stats in batches.items():
            lines.append(
                f"| {backend} | {name.replace('batch', '')} | {stats['median_ms']:.2f} | "
                f"{stats['p95_ms']:.2f} | {stats['throughput_img_s']:.1f} |"
            )
    table = "\n".join(lines)

    if "torch" in results["backends"] and "onnx" in results["backends"]:
        speedups = {}
        for name in results["backends"]["torch"]:
            torch_ms = results["backends"]["torch"][name]["median_ms"]
            onnx_ms = results["backends"]["onnx"][name]["median_ms"]
            speedups[name] = round(torch_ms / onnx_ms, 2)
        results["speedup_onnx_vs_torch"] = speedups
        table += "\n\n**ONNX tezlashuvi:** " + ", ".join(
            f"{k.replace('batch', 'batch=')} -> {v}x" for k, v in speedups.items()
        )
        print("\nONNX tezlashuvi:", speedups)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    print(f"\n-> {args.out}")
    print(f"-> {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
