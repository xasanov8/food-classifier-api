# O'zbek taomlari klassifikatori — Production API

FastAPI + ONNX Runtime + Docker. Notebook'dagi modelni HTTP xizmatiga aylantirish.

[`uzbek-food-classifier`](https://github.com/xasanov8/uzbek-food-classifier) loyihasida
o'qitilgan ResNet18 modelini oladi, ONNX'ga eksport qiladi va konteynerda
xizmat sifatida ishga tushiradi.

```
rasm -> preprocessing (NumPy) -> ONNX Runtime -> softmax -> chegara -> JSON
```

---

## Tez boshlash

### Docker bilan

```bash
docker compose up --build
```

Brauzerda `http://localhost:8000` — rasm yuklash sahifasi.
API hujjati: `http://localhost:8000/docs`

> **Eslatma:** loyiha ishlab chiqilgan mashinada WSL2 o'rnatilmagani uchun
> Docker qurilishi **ishga tushirib ko'rilmagan**. Dockerfile va
> `docker-compose.yml` standart ko'p bosqichli shablonga asoslangan,
> lekin obraz hajmi va ishga tushish vaqti o'lchanmagan. Quyidagi
> benchmark va testlar esa to'g'ridan-to'g'ri xost mashinada
> o'tkazilgan va haqiqiy.

### Docker'siz

```bash
pip install -r requirements-dev.txt
python scripts/export_onnx.py --checkpoint ../uzbek-food-classifier/reports/resnet18/best.pt
uvicorn app.main:app --reload
```

---

## API

| Method | Endpoint | Tavsif |
|---|---|---|
| `GET` | `/` | Rasm yuklash sahifasi |
| `GET` | `/health` | Tiriklik tekshiruvi (Docker healthcheck) |
| `GET` | `/classes` | Model biladigan sinflar |
| `POST` | `/predict` | Bitta rasm |
| `POST` | `/predict/batch` | Bir nechta rasm (ko'pi bilan 16 ta) |

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@osh.jpg"
```

```json
{
  "filename": "osh.jpg",
  "label": "osh",
  "confidence": 0.9614,
  "is_confident": true,
  "message": null,
  "top_k": [
    {"label": "osh", "score": 0.9614},
    {"label": "lagmon", "score": 0.0183},
    {"label": "manti", "score": 0.0094}
  ],
  "scores": { "...": "barcha sinflar bo'yicha to'liq taqsimot" },
  "latency_ms": 18.4,
  "backend": "onnx"
}
```

---

## Nima uchun ONNX

ONNX Runtime bir xil modelni bir xil kirishda PyTorch'dan tezroq bajaradi,
chunki u grafni oldindan optimallashtiradi: `Conv + BatchNorm` birlashtiriladi,
o'lik tugunlar tashlanadi, xotira joylashuvi qayta rejalashtiriladi.

O'lchov: RTX 3050 Ti laptop, **CPU inference** (production konteynerida GPU
bo'lmaydi), 150 ta chaqiruv + 20 ta warmup.

| Backend | Batch | Median | p95 | O'tkazuvchanlik |
|---|---|---|---|---|
| PyTorch | 1 | 31.76 ms | 33.57 ms | 31.5 rasm/s |
| **ONNX Runtime** | 1 | **19.46 ms** | 22.27 ms | **51.4 rasm/s** |
| PyTorch | 8 | 190.64 ms | 214.84 ms | 42.0 rasm/s |
| **ONNX Runtime** | 8 | **159.52 ms** | 174.01 ms | **50.2 rasm/s** |

**Tezlashuv: batch=1 da 1.63x, batch=8 da 1.20x.**

Batch=1 da farq kattaroq — aynan shu API uchun muhim rejim, chunki
foydalanuvchi bittadan rasm yuklaydi. Katta batch'da PyTorch o'zining
overhead'ini ko'p ish ustiga yoyadi va farq qisqaradi.

Eksport ekvivalentligi tekshiriladi — `scripts/export_onnx.py` ikkala modelni
tasodifiy kirishlarda solishtiradi:

```
PyTorch va ONNX orasidagi maksimal farq: 1.79e-06
Tekshiruv o'tdi: ikkala model bir xil natija beradi.
```

1.79e-06 — fp32 hisob-kitobning tabiiy tafovuti. Bundan kattasi eksport
xatosidan darak berardi va skript xato bilan to'xtardi.

Ikkinchi yutuq — **bog'liqliklar hajmi**: `model.onnx` 42.6 MB, va
konteynerga PyTorch umuman kerak emas. `torch` + `torchvision`
g'ildiraklari ~800 MB joy egallaydi; `onnxruntime` ~15 MB.

O'lchovni qayta ishlab ko'rish:

```bash
python scripts/benchmark.py --runs 150
```

---

## Production'ga oid qarorlar

### 1. Obrazda PyTorch yo'q

`requirements.txt` da `torch` yo'q — faqat `onnxruntime`, `fastapi`, `Pillow`,
`numpy`. PyTorch faqat `requirements-dev.txt` da: u ONNX eksporti, benchmark
va parity testi uchun kerak, xizmatning o'zi uchun emas.

Buning narxi: preprocessing'ni `torchvision` siz, qo'lda yozish kerak
bo'ldi ([`app/preprocessing.py`](app/preprocessing.py)).

### 2. Preprocessing parity testi

Bu loyihaning eng katta xavfi — xizmatdagi rasm tayyorlash o'qitishdagidan
farq qilib qolishi. Bunday xato exception bermaydi: model shunchaki
jimgina yomonroq bashorat qiladi va buni production'da sezish deyarli imkonsiz.

[`scripts/verify_parity.py`](scripts/verify_parity.py) haqiqiy rasmlarda
`torchvision` quvurini va qo'lda yozilgan NumPy quvurini yonma-yon
solishtiradi:

```bash
python scripts/verify_parity.py --images ../uzbek-food-classifier/data/clean
```

Natija:

```
Tekshirilgan rasmlar : 120
Maksimal farq        : 0.000e+00
OK: xizmatdagi preprocessing o'qitishdagi bilan bir xil.
```

**Bu test darhol haqiqiy xatoni topdi.** Birinchi versiyada `Resize` uzun
tomonni `round()` bilan yaxlitlagan edi, torchvision esa `int()` bilan
**kesadi**:

```python
new_h = int(target * height / width)     # torchvision — kesadi
new_h = int(round(target * height / width))  # mening xatom — yaxlitlaydi
```

Farq bir piksel. Lekin undan keyingi `CenterCrop` butunlay boshqa sohani
kesib oladi — natijadagi tenzorlar orasidagi maksimal farq **2.15** edi.
Test bo'lmaganida bu production'ga jimgina o'tib ketardi.

### 3. Softmax hech qachon "bilmayman" demaydi

Bu xizmatning eng ko'p uchraydigan amaliy muammosi. Softmax har doim
yig'indisi 1 ga teng taqsimot qaytaradi — model ro'yxatda yo'q taomni
(yoki umuman taom bo'lmagan rasmni) ko'rsa ham, ballarni mavjud sinflar
orasida taqsimlaydi va ko'pincha **yuqori ishonch bilan xato aytadi**.

Yechim — ishonch chegarasi. `max(softmax) < T` bo'lsa javob
`is_confident: false` bilan keladi va foydalanuvchiga izoh beriladi:

```json
{
  "label": "somsa",
  "confidence": 0.31,
  "is_confident": false,
  "message": "Ishonch past. Bu taom modelga tanish bo'lmasligi yoki rasm noaniq bo'lishi mumkin...",
  "top_k": [
    {"label": "somsa", "score": 0.31},
    {"label": "qatlama", "score": 0.28},
    {"label": "non", "score": 0.17}
  ]
}
```

Chegara qiymati ko'zdan taxmin qilinmaydi —
[`scripts/calibrate_threshold.py`](scripts/calibrate_threshold.py) uni test
to'plamida kalibrlaydi. Har bir chegara uchun ikkita raqam hisoblanadi:

| Metrika | Ma'nosi |
|---|---|
| `coverage` | nechta bashorat "ishonchli" deb qoldi |
| `selective_accuracy` | shu qoldirilganlar orasidagi aniqlik |

Chegara juda past bo'lsa noma'lum taomlar o'tib ketadi; juda baland
bo'lsa to'g'ri bashoratlar ham "bilmayman" ga aylanadi. Skript berilgan
minimal qamrovni saqlagan holda tanlangan aniqlikni maksimallashtiradi va
qiymatni `models/labels.json` ga yozadi.

```bash
python scripts/calibrate_threshold.py --write
```

Ustuvorlik tartibi: `MIN_CONFIDENCE` muhit o'zgaruvchisi -> model bilan
kelgan kalibrlangan qiymat -> sukut (0.45).

### 4. Model bir marta yuklanadi

Model FastAPI `lifespan` da yuklanadi va `state` da saqlanadi. Har so'rovda
qayta yuklash ~300 ms qo'shar va xotirani behuda sarflar edi.
Startupda `warmup()` bir necha bo'sh chaqiruv qiladi — birinchi haqiqiy
so'rov sekin bo'lmasligi uchun.

### 5. Xavfsizlik va chegaralar

| Chegara | Qiymat | Sabab |
|---|---|---|
| Yuklash hajmi | 8 MB | Cheklovsiz upload xizmatni xotira bilan yiqitadi |
| Batch hajmi | 16 ta rasm | Bitta so'rov CPU'ni butunlay egallamasligi uchun |
| ONNX oqimlari | 2 | Konteynerda odatda 1-2 vCPU bo'ladi |
| Konteyner foydalanuvchisi | `appuser` (uid 10001) | root'dan ishlamaslik |

`content-type` tekshiriladi (415), buzilgan rasm 400 qaytaradi,
model yuklanmagan bo'lsa 503.

---

## Loyiha tuzilishi

```
app/
  main.py           FastAPI endpointlari
  inference.py      Classifier — ONNX va PyTorch bitta interfeys ortida
  preprocessing.py  torchvision quvurining NumPy takrori
  schemas.py        Pydantic modellari (OpenAPI hujjati shu yerdan)
  torch_model.py    Arxitektura ta'rifi (faqat BACKEND=torch uchun)
scripts/
  export_onnx.py    checkpoint -> ONNX + ekvivalentlik tekshiruvi
  benchmark.py      PyTorch vs ONNX latency
  calibrate_threshold.py  ishonch chegarasini test to'plamida tanlash
  verify_parity.py  preprocessing parity testi
static/index.html   drag & drop interfeys
tests/test_api.py   API va preprocessing testlari
Dockerfile          ko'p bosqichli qurilish
```

---

## Testlar

```bash
pytest -q
```

```
12 passed
```

### Uchdan-uchgacha tekshiruv

Xizmat haqiqiy rasmlarda ishga tushirib ko'rilgan (`uvicorn app.main:app`):

```
$ curl -s localhost:8000/health
{"status":"ok","backend":"onnx","model":"resnet18","classes":5}

$ curl -s -X POST localhost:8000/predict -F "file=@osh.jpg"
{"filename":"osh.jpg","label":"osh","confidence":0.9361,
 "scores":{"chuchvara":0.0153,"lagmon":0.0113,"manti":0.0089,
           "osh":0.9361,"somsa":0.0285},
 "latency_ms":18.92,"backend":"onnx"}

$ curl -s -X POST localhost:8000/predict -F "file=@somsa.jpg"
{"label":"somsa","confidence":0.9976, ... "latency_ms":16.9}
```

Web interfeys ham brauzerda tekshirilgan: rasm yuklanganda `/predict`
chaqiriladi, natija va besh sinf bo'yicha ehtimolliklar ko'rsatiladi
(`osh`, ishonch 93.6%, 16.9 ms, backend `onnx`).

Testlar mock ishlatmaydi — haqiqiy ONNX model yuklanadi. Tekshiriladi:
softmax haqiqiy ehtimollik taqsimoti ekani, bir xil rasm bir xil natija
berishi (`eval()` rejimi buzilmaganini bildiradi), noto'g'ri kirishlar
to'g'ri HTTP kod qaytarishi, turli nisbatdagi rasmlar cho'zilmasdan
kesilishi.

---

## Konfiguratsiya

| O'zgaruvchi | Sukut | Tavsif |
|---|---|---|
| `BACKEND` | `onnx` | `onnx` yoki `torch` |
| `MODELS_DIR` | `models` | Model fayllari papkasi |
| `ORT_THREADS` | `2` | ONNX Runtime oqimlari soni |
| `MAX_UPLOAD_BYTES` | `8388608` | Yuklash chegarasi |
| `MAX_BATCH` | `16` | Batch so'rovdagi rasmlar chegarasi |
| `MIN_CONFIDENCE` | kalibrlangan | Bundan past ishonch `is_confident: false` |

---

## Bog'liq loyihalar

- [uzbek-food-classifier](https://github.com/xasanov8/uzbek-food-classifier) — bu yerdagi model qayerdan kelgani
- [uzbek-sentiment-xlmr](https://github.com/xasanov8/uzbek-sentiment-xlmr) — o'zbek tilida NLP fine-tuning
