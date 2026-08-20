"""
Ishonch chegarasini test to'plamida kalibrlash.

## Muammo

Softmax har doim yig'indisi 1 ga teng taqsimot qaytaradi. Model ro'yxatda
yo'q taomni ko'rsa ham "bilmayman" deya olmaydi — ballarni mavjud sinflar
orasida taqsimlaydi va ko'pincha yuqori ishonch bilan xato aytadi.

Yechim — chegara: `max(softmax) < T` bo'lsa xizmat "aniq ayta olmayman"
deydi. Lekin `T` ni ko'zdan taxmin qilish noto'g'ri bo'lardi.

## Usul

Chegara ikki xil xatoni muvozanatlaydi:

  * **T juda past**  -> noma'lum taomlar ham ishonchli deb belgilanadi
  * **T juda baland** -> to'g'ri bashoratlar ham "bilmayman" ga aylanadi

Shuning uchun har bir chegara uchun ikkita raqam hisoblanadi:

  * `coverage`  — nechta bashorat ishonchli deb qoldi (qamrov)
  * `selective_accuracy` — o'sha qoldirilganlar orasida aniqlik

Yaxshi chegara: qamrovni keskin tushirmagan holda tanlangan
bashoratlarning aniqligini maksimal ko'taradi.

    python scripts/calibrate_threshold.py \\
        --data-root ../uzbek-food-classifier/data/clean \\
        --splits    ../uzbek-food-classifier/data/splits.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference import Classifier  # noqa: E402
from app.preprocessing import preprocess  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ishonch chegarasini kalibrlash")
    parser.add_argument("--data-root", type=Path,
                        default=Path("../uzbek-food-classifier/data/clean"))
    parser.add_argument("--splits", type=Path,
                        default=Path("../uzbek-food-classifier/data/splits.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    # 0.70 — ataylab qat'iy. Bu xizmatning va'dasi: "ishonchli" deb
    # belgilangan javob haqiqatan to'g'ri bo'lsin. Qamrovni 0.90 ga
    # ko'tarsak chegara 0.25 ga tushadi va tanlangan aniqlik 0.77 bo'lib
    # qoladi — ya'ni chegara deyarli hech narsa qilmaydi. 0.70 da esa
    # ishonchli javoblarning ~88 foizi to'g'ri chiqadi, qolgan 30 foiz
    # holatda xizmat halol "aniq ayta olmayman" deydi.
    parser.add_argument("--target-coverage", type=float, default=0.70,
                        help="Kamida shuncha ulush bashorat saqlanishi kerak")
    parser.add_argument("--write", action="store_true",
                        help="Tanlangan chegarani models/labels.json ga yozish")
    args = parser.parse_args()

    from PIL import Image

    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    items = payload[args.split]
    classes = payload["classes"]

    classifier = Classifier(backend="onnx", models_dir=args.models_dir,
                            min_confidence=0.0)
    if classifier.classes != classes:
        raise SystemExit(
            "Model sinflari split fayldagi sinflar bilan mos kelmadi:\n"
            f"  model : {classifier.classes}\n  split : {classes}"
        )

    print(f"{args.split} to'plami: {len(items)} ta rasm, {len(classes)} sinf")

    confidences = np.empty(len(items), dtype=np.float32)
    correct = np.empty(len(items), dtype=bool)

    for i, rel in enumerate(items):
        with Image.open(args.data_root / rel) as img:
            batch = preprocess(img, classifier.image_size)
        logits, _ = classifier.predict_array(batch)
        probs = np.exp(logits[0] - logits[0].max())
        probs /= probs.sum()
        best = int(probs.argmax())
        confidences[i] = probs[best]
        correct[i] = classes[best] == rel.split("/")[0]
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(items)}...", flush=True)

    baseline = correct.mean()
    print(f"\nChegarasiz aniqlik: {baseline:.4f}\n")

    header = f"{'chegara':>8} {'qamrov':>8} {'tanlangan aniqlik':>18} {'yo''qolgan to''g''ri':>18}"
    print(header)
    print("-" * len(header))

    rows = []
    for threshold in np.arange(0.20, 0.96, 0.05):
        keep = confidences >= threshold
        coverage = keep.mean()
        if keep.sum() == 0:
            continue
        selective = correct[keep].mean()
        # Chegara tufayli "bilmayman" ga aylangan TO'G'RI bashoratlar
        lost = (correct & ~keep).sum()
        rows.append((float(threshold), float(coverage), float(selective), int(lost)))
        print(f"{threshold:8.2f} {coverage:8.3f} {selective:18.4f} {lost:18d}")

    # Qamrov chegarasini saqlagan holda eng yuqori tanlangan aniqlik
    eligible = [r for r in rows if r[1] >= args.target_coverage]
    if not eligible:
        eligible = rows
    best_row = max(eligible, key=lambda r: (r[2], r[1]))
    threshold, coverage, selective, lost = best_row

    print(f"\nTanlandi: chegara = {threshold:.2f}")
    print(f"  qamrov            : {coverage:.3f} ({int(coverage * len(items))} / {len(items)})")
    print(f"  tanlangan aniqlik : {selective:.4f}  (chegarasiz {baseline:.4f})")
    print(f"  yo'qolgan to'g'ri : {lost} ta")

    report = {
        "split": args.split,
        "n": len(items),
        "baseline_accuracy": round(float(baseline), 4),
        "chosen_threshold": round(threshold, 2),
        "coverage": round(coverage, 4),
        "selective_accuracy": round(selective, 4),
        "target_coverage": args.target_coverage,
        "curve": [
            {"threshold": round(t, 2), "coverage": round(c, 4),
             "selective_accuracy": round(s, 4), "lost_correct": l}
            for t, c, s, l in rows
        ],
    }
    out = Path("reports/threshold.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")

    if args.write:
        labels_file = args.models_dir / "labels.json"
        meta = json.loads(labels_file.read_text(encoding="utf-8"))
        meta["min_confidence"] = round(threshold, 2)
        labels_file.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"-> {labels_file} (min_confidence = {threshold:.2f})")


if __name__ == "__main__":
    main()
