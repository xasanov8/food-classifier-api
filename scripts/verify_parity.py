"""
Preprocessing tengligini tekshirish: torchvision vs qo'lda yozilgan NumPy.

Bu skript loyihaning eng katta xavfini yopadi. `app/preprocessing.py`
torchvision quvurini qo'lda takrorlaydi — agar bironta detal (resize
qiyofasi, crop markazi, normalizatsiya tartibi) noto'g'ri bo'lsa, xizmat
xato bermaydi, shunchaki jimgina yomonroq bashorat qiladi. Bunday xatoni
production'da topish deyarli imkonsiz.

    python scripts/verify_parity.py --images ../uzbek-food-classifier/data/clean

torch/torchvision faqat shu tekshiruv uchun kerak — production obraziga
kirmaydi.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing import preprocess  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing parity testi")
    parser.add_argument("--images", type=Path, required=True,
                        help="Rasmlar papkasi (rekursiv qidiriladi)")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    import json

    from torchvision import transforms

    from app.preprocessing import DEFAULT_IMAGE_SIZE, RESIZE_RATIO

    # O'lchamni modeldan olamiz — parity testining o'zi eskirgan
    # konstantani tekshirib o'tirishi bema'nilik bo'lardi.
    labels_file = args.models_dir / "labels.json"
    if labels_file.exists():
        image_size = int(json.loads(labels_file.read_text(encoding="utf-8"))
                         .get("image_size", DEFAULT_IMAGE_SIZE))
    else:
        image_size = DEFAULT_IMAGE_SIZE
    print(f"Rasm o'lchami            : {image_size}")

    reference = transforms.Compose([
        transforms.Resize(int(image_size * RESIZE_RATIO)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    paths = sorted(args.images.rglob("*.jpg"))[: args.limit]
    if not paths:
        print(f"Rasm topilmadi: {args.images}")
        sys.exit(1)

    worst = 0.0
    worst_path = None
    for path in paths:
        with Image.open(path) as img:
            img = img.convert("RGB")
            expected = reference(img).unsqueeze(0).numpy()
            actual = preprocess(img, image_size)

        diff = float(np.abs(expected - actual).max())
        if diff > worst:
            worst, worst_path = diff, path

    print(f"Tekshirilgan rasmlar     : {len(paths)}")
    print(f"Maksimal farq            : {worst:.3e}")
    print(f"Eng katta farqli rasm    : {worst_path}")
    print(f"Ruxsat etilgan chegara   : {args.tolerance:.1e}")

    if worst > args.tolerance:
        print("\nXATO: preprocessing torchvision bilan mos kelmayapti.")
        sys.exit(1)
    print("\nOK: xizmatdagi preprocessing o'qitishdagi bilan bir xil.")


if __name__ == "__main__":
    main()
