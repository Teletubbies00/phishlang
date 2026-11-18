
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]
