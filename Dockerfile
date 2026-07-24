FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        -r /app/backend/requirements.txt

COPY backend /app/backend

CMD ["sh", "-c", "uvicorn backend.index:app --host 0.0.0.0 --port ${PORT:-8000}"]