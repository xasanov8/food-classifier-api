# ---------- 1-bosqich: bog'liqliklarni qurish ----------
# Ko'p bosqichli qurilish: wheel'lar alohida bosqichda tayyorlanadi,
# yakuniy obrazga kompilyator ham, build keshi ham tushmaydi.
FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---------- 2-bosqich: ishchi obraz ----------
FROM python:3.12-slim

# PyTorch bu obrazda YO'Q. requirements.txt faqat onnxruntime, FastAPI,
# Pillow va numpy'ni o'z ichiga oladi. Taqqoslash uchun: torch+torchvision
# g'ildiraklarining o'zi ~800 MB, onnxruntime esa ~15 MB.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BACKEND=onnx \
    MODELS_DIR=/app/models \
    ORT_THREADS=2

WORKDIR /app

# root'dan ishlamaslik — konteyner buzilganda zarar chegaralanadi.
RUN useradd --create-home --uid 10001 appuser

COPY app/ ./app/
COPY static/ ./static/
COPY models/ ./models/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck: orkestrator konteyner tirikligini shu orqali biladi.
# start-period model yuklanishi uchun vaqt beradi.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
