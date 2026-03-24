# ──────────────────────────────────────────────────────────────────────────────
# DeepShield — Deepfake Detection System
# Base: python:3.10-slim  (CPU-only, no GPU required)
# Ports: 5000 (Flask API) | 8501 (Streamlit Dashboard)
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /deepshield

# ── Python dependencies (installed before copying source for layer caching) ───
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# ── Create directories used at runtime ────────────────────────────────────────
RUN mkdir -p static/uploads

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 5000
EXPOSE 8501

# ── Healthcheck — polls the Flask /health endpoint ───────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# ── Default command: run the Streamlit dashboard ──────────────────────────────
# Override with: docker run ... python app/api.py  (to start Flask instead)
CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
