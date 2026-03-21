FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY spuncast_ml ./spuncast_ml

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY . .

ENTRYPOINT ["spuncast-ml"]
CMD ["pipeline"]

