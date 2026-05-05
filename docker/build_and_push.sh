#!/usr/bin/env bash
set -euo pipefail

# Build and push Docker image for PyTorch cu117
# Requires: docker login (DockerHub credentials configured)

IMAGE_NAME="andryusha2006/pytorch-cu117"
IMAGE_TAG="2.0.0"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image: $FULL_IMAGE"
docker build \
  -t "$FULL_IMAGE" \
  -f "$(dirname "$0")/Dockerfile.pytorch-cu117" \
  "$(dirname "$0")/.."

echo "Pushing to DockerHub: $FULL_IMAGE"
docker push "$FULL_IMAGE"

echo "✓ Successfully pushed $FULL_IMAGE"
echo "  You can now use this image in ClearML config:"
echo "    clearml:"
echo "      docker_image: \"$FULL_IMAGE\""
