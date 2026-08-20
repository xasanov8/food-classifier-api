# O'zbek taomlari klassifikatori — Production API

FastAPI + ONNX Runtime + Docker. Notebook'dagi modelni HTTP xizmatiga aylantirish.

[`uzbek-food-classifier`](https://github.com/xasanov8/uzbek-food-classifier) loyihasida
o'qitilgan ResNet18 modelini oladi (16 ta o'zbek taomi, 288px), ONNX'ga
eksport qiladi va konteynerda xizmat sifatida ishga tushiradi.

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

O'lchov 288x288 kirishda (joriy model), 100 chaqiruv + 15 warmup:

| Backend | Batch | Median | p95 | O'tkazuvchanlik |
|---|---|---|---|---|
| PyTorch | 1 | 42.34 ms | 46.00 ms | 23.6 rasm/s |
| **ONNX Runtime** | 1 | **32.14 ms** | 33.26 ms | **31.1 rasm/s** |
| PyTorch | 8 | 290.49 ms | 313.54 ms | 27.5 rasm/s |
| **ONNX Runtime** | 8 | **251.51 ms** | 263.92 ms | **31.8 rasm/s** |

**Tezlashuv: batch=1 da 1.32x, batch=8 da 1.15x.**

224px li 5 sinfli modelda tezlashuv kattaroq edi (1.63x / 19.46 ms).
Rezolyutsiya 288 ga ko'tarilgach ikkala backend ham sekinlashdi va
ONNX'ning nisbiy ustunligi qisqardi — katta tenzorlarda graf
optimizatsiyasining ulushi kamayadi.

Batch=1 da farq kattaroq — aynan shu API uchun muhim rejim, chunki
foydalanuvchi bittadan rasm yuklaydi. Katta batch'da PyTorch o'zining
overhead'ini ko'p ish ustiga yoyadi va farq qisqaradi.

Eksport ekvivalentligi tekshiriladi — `scripts/export_onnx.py` ikkala modelni
tasodifiy kirishlarda solishtiradi:

```
Arxitektura: resnet18, o'lcham: 288, sinflar (16): [...]
PyTorch va ONNX orasidagi maksimal farq: 3.58e-06
Tekshiruv o'tdi: ikkala model bir xil natija beradi.
```

3.58e-06 — fp32 hisob-kitobning tabiiy tafovuti. Bundan kattasi eksport
xatosidan darak berardi va skript xato bilan to'xtardi.

Ikkinchi yutuq — **bog'liqliklar hajmi**: `model.onnx` 42.6 MB, va
konteynerga PyTorch umuman kerak emas. `torch` + `torchvision`
g'ildiraklari ~800 MB joy egallaydi; `onnxruntime` ~15 MB.

O'lchovni qayta ishlab ko'rish:

```bash
python scripts/benchmark.py --runs 100
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
Rasm o'lchami        : 288
Tekshirilgan rasmlar : 120
Maksimal farq        : 0.000e+00
OK: xizmatdagi preprocessing o'qitishdagi bilan bir xil.
```

Rasm o'lchami ham `models/labels.json` dan o'qiladi. Ilgari u kodda
`224` deb qotirib yozilgan edi — model 288px da qayta o'qitilgach, bu
xizmatni jimgina noto'g'ri kesib berishga majbur qilardi. Aynan
shu turdagi xato uchun parity testi yozilgan.

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
  "confidence": 0.27,
  "is_confident": false,
  "message": "Ishonch past. Bu taom modelga tanish bo'lmasligi yoki rasm noaniq bo'lishi mumkin...",
  "top_k": [
    {"label": "somsa", "score": 0.27},
    {"label": "qatlama", "score": 0.22},
    {"label": "non", "score": 0.14}
  ]
}
```

Bu muammoni foydalanuvchi topdi: kovatak (tok bargidagi dolma) rasmini
yubordi, model uni ishonch bilan boshqa taom deb atadi.

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

Joriy model uchun egri chiziq (625 ta test rasmi, chegarasiz aniqlik 0.7296):

| Chegara | Qamrov | Tanlangan aniqlik |
|---|---|---|
| 0.25 | 0.930 | 0.7676 |
| 0.35 | 0.830 | 0.8112 |
| **0.45** | **0.704** | **0.8773** |
| 0.55 | 0.610 | 0.9081 |
| 0.70 | 0.474 | 0.9392 |

**0.45 tanlandi.** Ya'ni xizmat "ishonchli" desa, javoblarning
**87.7 foizi** to'g'ri — chegarasiz bu 73 foiz edi. Buning evaziga
rasmlarning 30 foizida xizmat halol "aniq ayta olmayman" deydi.

Chegarani yanada ko'tarish aniqlikni oshiradi, lekin qamrov yarmiga
tushadi — foydalanuvchi uchun bu yomonroq tajriba.

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

Xizmat haqiqiy rasmlarda ishga tushirib ko'rilgan. Har bir sinfdan bitta
test rasmi yuborildi:

```
chuchvara   -> OK   chuchvara   0.783  ishonchli
dimlama     -> OK   dimlama     0.422  ishonchsiz
hasip       -> OK   hasip       0.391  ishonchsiz
kovatak     -> XATO mastava     0.443  ishonchsiz
lagmon      -> XATO somsa       0.254  ishonchsiz
manti       -> OK   manti       0.526  ishonchli
mastava     -> OK   mastava     0.874  ishonchli
non         -> OK   non         0.852  ishonchli
norin       -> OK   norin       0.954  ishonchli
osh         -> OK   osh         0.371  ishonchsiz
qatlama     -> XATO somsa       0.270  ishonchsiz
shashlik    -> XATO hasip       0.229  ishonchsiz
shurva      -> OK   shurva      0.392  ishonchsiz
somsa       -> OK   somsa       0.596  ishonchli
sumalak     -> OK   sumalak     0.778  ishonchli
tuxum_barak -> OK   tuxum_barak 0.306  ishonchsiz

12/16 to'g'ri, 9 tasida model ishonchsiz
```

Diqqat qiling: **to'rtta xatoning hammasi "ishonchsiz" deb belgilangan**,
va ishonchli deb belgilangan yettita javobning hammasi to'g'ri. Chegara
aynan shuning uchun qo'yilgan.

Web interfeys ham brauzerda tekshirilgan: rasm yuklanganda `/predict`
chaqiriladi, ishonch past bo'lsa "Aniq ayta olmayman" holati va eng
yaqin 5 variant ko'rsatiladi.

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
