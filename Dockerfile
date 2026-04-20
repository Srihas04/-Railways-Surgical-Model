# ─────────────────────────────────────────────────────────────────────
#  Surgical Guardian v4 — Ultra-slim Railway image
#  Target: < 2.5 GB  (vs 6.5 GB before)
#  Strategy:
#    1. python:3.11-slim  (Debian bookworm, ~50 MB)
#    2. CPU-only torch+torchvision installed FIRST via pip index override
#    3. ultralytics installed AFTER (won't re-download CUDA torch)
#    4. opencv-python-headless (no Qt/GTK)
#    5. Clean pip cache + apt cache in the SAME layer
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps needed by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: CPU-only PyTorch FIRST ───────────────────────────────────
# This is the critical step — we pin to the +cpu build so pip never
# fetches the 2.5 GB CUDA variant that ultralytics would otherwise pull.
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchvision==0.18.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Everything else ───────────────────────────────────────────
# ultralytics will detect torch already installed and skip re-downloading
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 3: App files ─────────────────────────────────────────────────
COPY surgical_guardian_web.py .
COPY best.pt .

# ── Step 4: Runtime ───────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=best.pt
ENV VIDEO_SOURCE=0
ENV CONF_THRESH=0.30
ENV FRAME_WIDTH=640
ENV FRAME_HEIGHT=480

EXPOSE 8080

CMD ["sh", "-c", "gunicorn surgical_guardian_web:app --workers 1 --threads 4 --bind 0.0.0.0:$PORT --timeout 120 --preload"]
