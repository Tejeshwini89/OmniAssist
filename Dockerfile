FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY data ./data
COPY evaluation ./evaluation
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["sh", "-c", "if [ ! -f artifacts/faiss_index/index.faiss ]; then python -m src.omniassist.ingest; fi && uvicorn src.omniassist.api:app --host 0.0.0.0 --port 8000"]
