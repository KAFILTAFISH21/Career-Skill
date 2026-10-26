# ---------- Stage 1: train (Sentence-BERT embeddings + XGBoost) ----------
FROM python:3.11-slim AS train
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
COPY requirements.txt requirements-train.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-train.txt
COPY . .
RUN python -m scripts.train

# ---------- Stage 2: slim runtime (FastAPI + dashboard) ----------
FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PORT=8000
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app app
COPY frontend frontend
COPY data/taxonomy data/taxonomy
COPY --from=train /app/artifacts artifacts
COPY --from=train /app/reports reports
RUN useradd -m appuser
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/api/health')"
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
