#!/usr/bin/env bash
# .devcontainer/build-and-push.sh
# Build the aitchison-flow training image and push to DockerHub.
#
# Usage:
#   DOCKERHUB_REPO=yourname/aitchison-flow ./build-and-push.sh [TAG]
#
# TAG defaults to the short git SHA so every build is traceable.
set -euo pipefail
REPO="${DOCKERHUB_REPO:-mvonsiebenth/aitchinson-flow}"
TAG="${1:-$(git rev-parse --short HEAD)}"
FULL_TAG="$REPO:$TAG"

echo "==> Building $FULL_TAG"
BUILD_CONTEXT="$(git rev-parse --show-toplevel)"
DOCKERFILE_PATH="$(dirname "$0")/Dockerfile"

if docker buildx version >/dev/null 2>&1; then
    docker buildx build \
        --platform linux/amd64 \
        --file "$DOCKERFILE_PATH" \
        --tag "$FULL_TAG" \
        --tag "$REPO:latest" \
        --push \
        "$BUILD_CONTEXT"
else
    echo "==> buildx not available; using docker build + docker push"
    docker build \
        --file "$DOCKERFILE_PATH" \
        --tag "$FULL_TAG" \
        --tag "$REPO:latest" \
        "$BUILD_CONTEXT"
    docker push "$FULL_TAG"
    docker push "$REPO:latest"
fi

echo "==> Done. Image available as:"
echo "    $FULL_TAG"
echo "    $REPO:latest"
