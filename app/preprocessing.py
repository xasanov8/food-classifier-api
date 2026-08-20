"""
Rasmni tayyorlash — torch'siz.

Bu faylning butun mazmuni bitta jumlada: PRODUCTION'DAGI PREPROCESSING
O'QITISHDAGIDAN BIR PIKSEL HAM FARQ QILMASLIGI KERAK.

O'qitishda torchvision ishlatilgan:
    Resize(255) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet)

Bu yerda xuddi shu ketma-ketlik PIL + NumPy bilan takrorlangan. Nima uchun
torchvision'ni chaqirmaymiz: shunda Docker obrazida PyTorch umuman kerak
emas (~800 MB o'rniga ~60 MB). Buning evaziga preprocessing'ni qo'lda
to'g'ri yozish majburiyati tug'iladi — `scripts/verify_parity.py` aynan
shuni tekshiradi.

Muhim detallar:
  * Resize(int) — KICHIK tomonni shu qiymatga keltiradi, nisbatni saqlagan holda
  * torchvision PIL rasmga PIL'ning o'z resize'ini qo'llaydi, shuning uchun
    Image.BILINEAR bit-darajasida bir xil natija beradi
  * ToTensor() 255 ga bo'ladi va HWC -> CHW ga aylantiradi
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SIZE = 224
RESIZE_TO = int(IMAGE_SIZE * 1.14)  # = 255, o'qitishdagi qiymat bilan bir xil


def resize_shorter_side(image: Image.Image, target: int) -> Image.Image:
    """torchvision.transforms.Resize(target) ning aynan o'zi.

    DIQQAT — uzun tomon int() bilan KESILADI, yaxlitlanmaydi.
    torchvision manbasidagi hisob (_compute_resized_output_size):

        short, long = (w, h) if w <= h else (h, w)
        new_short, new_long = size, int(size * long / short)

    Bu yerda round() ishlatilsa, o'lchamlar hollarning taxminan yarmida
    1 piksel farq qiladi. Bir piksel arzimas ko'rinadi, lekin undan keyingi
    CenterCrop butunlay boshqa sohani kesib oladi va natijada tensor
    o'qitishdagidan sezilarli farq qiladi (scripts/verify_parity.py buni
    aynan shu tarzda aniqladi: maksimal farq 2.15 edi).
    """
    width, height = image.size
    if width <= height:
        new_w = target
        new_h = int(target * height / width)
    else:
        new_h = target
        new_w = int(target * width / height)
    return image.resize((new_w, new_h), Image.BILINEAR)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = int(round((width - size) / 2.0))
    top = int(round((height - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL rasm -> (1, 3, 224, 224) float32 NCHW massiv."""
    image = image.convert("RGB")
    image = resize_shorter_side(image, RESIZE_TO)
    image = center_crop(image, IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0        # ToTensor
    array = (array - IMAGENET_MEAN) / IMAGENET_STD             # Normalize
    array = array.transpose(2, 0, 1)                           # HWC -> CHW
    return np.ascontiguousarray(array[None, ...], dtype=np.float32)


def preprocess_bytes(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        return preprocess(image)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Barqaror softmax. Maksimalni ayirish overflow'ning oldini oladi."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)
