#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")" || exit 1

command -v docker >/dev/null 2>&1 || { echo "docker not found in PATH" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon not running / not accessible" >&2; exit 1; }

IMAGE_NAME="${IMAGE_NAME:-onnx-to-memryx}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
OUT_DIR="${OUT_DIR:-build}"
DOCKER_PROGRESS="${DOCKER_PROGRESS:-auto}"

rm -rf "$OUT_DIR" >/dev/null 2>&1 || true
mkdir -p "$OUT_DIR"

docker build --progress="$DOCKER_PROGRESS" -t "${IMAGE_NAME}:${IMAGE_TAG}" .

docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "./${OUT_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.tar"

# docker load -i "./${OUT_DIR}/${IMAGE_NAME}-${IMAGE_TAG}.tar"
# docker run -v ./memryx-deps:/app/memryx-deps -v ./artifacts:/app2 \
#   "${IMAGE_NAME}:${IMAGE_TAG}" /app2/model.zip /app2
