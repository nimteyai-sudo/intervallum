FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md README.ru.md ./
COPY src ./src
COPY scripts ./scripts
COPY content ./content
COPY docs ./docs
COPY examples ./examples

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

CMD ["uvicorn", "intervallum_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
