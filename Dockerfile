FROM python:3.12-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG PAPER_RAG_VERSION=0.1.0
ARG PAPER_RAG_REVISION=unknown

LABEL org.opencontainers.image.source="https://github.com/acemilee/metasurface-paper-RAG" \
      org.opencontainers.image.version="${PAPER_RAG_VERSION}" \
      org.opencontainers.image.revision="${PAPER_RAG_REVISION}" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY LICENSE THIRD_PARTY_NOTICES.md ./
RUN python -m pip install "pip==25.3"
ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=f82e2ae20c1545bb03997d1cc3143d94e14b800038669ee1aca45808a9acc338
RUN set -eu; \
    for attempt in 1 2 3; do \
        if python -c "import hashlib,pathlib,urllib.request; p=pathlib.Path('/tmp/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl'); request=urllib.request.Request('$TORCH_WHEEL_URL', headers={'User-Agent':'pip/25.3'}); p.write_bytes(urllib.request.urlopen(request, timeout=600).read()); actual=hashlib.sha256(p.read_bytes()).hexdigest(); assert actual == '$TORCH_WHEEL_SHA256', f'Torch wheel checksum mismatch: {actual}'"; then \
            break; \
        fi; \
        rm -f /tmp/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        sleep 5; \
    done
RUN python -m pip install /tmp/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl \
    && rm /tmp/torch-2.11.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install .

COPY alembic.ini ./
COPY alembic ./alembic
COPY src ./src
COPY scripts ./scripts

CMD ["python", "-m", "uvicorn", "paper_rag.main:app", "--host", "0.0.0.0", "--port", "8010"]
