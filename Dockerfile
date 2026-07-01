# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first (cached layer) using only the project metadata.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Non-root runtime user.
RUN useradd -m -u 1000 gateway
USER gateway

EXPOSE 8000
# GATEWAY_HOST/PORT and ACP_URL are supplied via environment at run time.
ENTRYPOINT ["acp-openai-gateway"]
