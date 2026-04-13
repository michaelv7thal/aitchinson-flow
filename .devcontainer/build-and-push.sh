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
docker buildx build \
    --platform linux/amd64 \
    --file "$(dirname "$0")/Dockerfile" \
    --tag "$FULL_TAG" \
    --tag "$REPO:latest" \
    --push \
    "$(git rev-parse --show-toplevel)"

echo "==> Done. Image available as:"
echo "    $FULL_TAG"
echo "    $REPO:latest"
