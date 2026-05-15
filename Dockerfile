FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY spuncast_ml ./spuncast_ml
COPY scripts ./scripts
COPY sql ./sql
COPY grafana ./grafana
COPY .env.example ./

RUN pip install --no-cache-dir --no-deps .

# Default image entrypoint: live scorer (docker-compose overrides for batch CLI).
ENTRYPOINT []
CMD ["python3", "scripts/score_heat_live.py"]
