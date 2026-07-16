#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

NO_BROWSER=0
BUILD_LOCAL=0
WAIT_TIMEOUT=1800
IMAGE=${PAPER_RAG_IMAGE:-ghcr.io/acemilee/metasurface-paper-rag:0.1.0}
BUILD_PROXY=${PAPER_RAG_BUILD_PROXY:-${HTTPS_PROXY:-}}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-browser)
            NO_BROWSER=1
            shift
            ;;
        --build-local)
            BUILD_LOCAL=1
            shift
            ;;
        --wait-timeout)
            [ "$#" -ge 2 ] || { echo "--wait-timeout requires seconds" >&2; exit 2; }
            WAIT_TIMEOUT=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

show_diagnostics() {
    printf '\n%s\n' "Compose status:"
    docker compose ps || true
    printf '\n%s\n' "Recent startup logs:"
    docker compose logs --tail 120 model-init migrate embedding worker api || true
}

command -v docker >/dev/null 2>&1 || {
    echo "Docker is not installed. Install and start Docker Desktop or Docker Engine, then retry." >&2
    exit 1
}
docker compose version >/dev/null 2>&1 || {
    echo "Docker Compose v2 is unavailable." >&2
    exit 1
}
docker info >/dev/null 2>&1 || {
    echo "Docker daemon is unavailable. Start Docker, then retry." >&2
    exit 1
}

if [ "$BUILD_LOCAL" -eq 1 ]; then
    if [ -n "$BUILD_PROXY" ]; then
        if ! docker build --tag paper-rag:local \
            --build-arg HTTP_PROXY="$BUILD_PROXY" \
            --build-arg HTTPS_PROXY="$BUILD_PROXY" .; then
            echo "Paper RAG local image build failed." >&2
            exit 1
        fi
    else
        if ! docker build --tag paper-rag:local .; then
            echo "Paper RAG local image build failed." >&2
            exit 1
        fi
    fi
    export PAPER_RAG_IMAGE=paper-rag:local
else
    if ! docker pull "$IMAGE"; then
        echo "Published image pull failed: $IMAGE. Retry or use --build-local." >&2
        exit 1
    fi
    export PAPER_RAG_IMAGE="$IMAGE"
fi

if ! docker compose up --detach --no-build --wait --wait-timeout "$WAIT_TIMEOUT"; then
    show_diagnostics
    exit 1
fi

if ! docker compose exec -T api python -c \
    "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8010/ready', timeout=10)); assert data['ready'] is True"; then
    show_diagnostics
    exit 1
fi

GUI_URL=http://127.0.0.1:8010
echo "GUI ready: http://127.0.0.1:8010"
if [ "$NO_BROWSER" -eq 0 ]; then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$GUI_URL" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
        open "$GUI_URL" >/dev/null 2>&1 || true
    fi
fi
