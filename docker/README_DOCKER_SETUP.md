# Docker Setup for ClearML Remote Workers

## Overview

The Docker image ensures that remote ClearML workers use **PyTorch 2.0.0+cu117** with CUDA 11.7 support.

## Quick Start

### 0. Run Without Docker

If you want ClearML to use the worker's native environment instead of a container, use the no-Docker config:

```bash
python3 train_grokking.py --config configs/grokking_mod_prime_113_nodocker.yaml
```

This config keeps `clearml.queue: gpu` but omits `docker_image`, so `task.execute_remotely()` runs on the worker's installed environment.

Use this when you want to test a different CUDA/PyTorch stack directly on the Windows worker.

### 1. Build and Push Image to DockerHub

**Prerequisites:**
- Docker installed locally
- DockerHub account
- `docker login` executed once

**Execute:**
```bash
cd docker
bash build_and_push.sh
```

This will:
- Build image `andryusha2006/pytorch-cu117:2.0.0` locally
- Push to DockerHub
- Print confirmation message

### 2. Verify Image on Remote Worker

After push completes, verify the remote worker can pull the image:
```bash
# On remote machine where clearml-agent runs
docker pull andryusha2006/pytorch-cu117:2.0.0
docker run --rm --gpus all andryusha2006/pytorch-cu117:2.0.0 python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Run Training with Docker

Once image is pushed and accessible:

```bash
# On local machine (where Task.init() creates the task)
python3 train_grokking.py --config configs/grokking_mod_prime_113.yaml
```

The config already specifies:
```yaml
clearml:
  queue: gpu
  docker_image: "andryusha2006/pytorch-cu117:2.0.0"
```

ClearML will:
1. Create a container from this image on the remote worker
2. Mount your repo code inside
3. Run training with PyTorch 2.0.0+cu117
4. GPU operations (CUDA kernels) will work on sm_61

---

## Dockerfile Details

- **Base:** `nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04`
- **Python:** 3.11
- **PyTorch:** 2.0.0+cu117
- **Dependencies:** numpy, matplotlib, clearml, pyyaml

---

## Troubleshooting

### "Image not found" on remote worker
- Ensure image is pushed to a registry accessible by the worker (DockerHub by default)
- Verify `docker pull andryusha2006/pytorch-cu117:2.0.0` works on the remote machine

### Worker still uses old PyTorch
- Ensure config has `docker_image` field
- Check ClearML logs: `docker ps` should show container name like `clearml-worker-...`

### Docker daemon not running on worker
- Contact your DevOps team; worker must have Docker installed and socket accessible to clearml-agent

---

## Custom Registry

To use a different registry (e.g., private registry, AWS ECR):

1. Edit `docker/build_and_push.sh` and update `IMAGE_NAME`
2. Ensure remote worker is authenticated to that registry
3. Update `configs/grokking_mod_prime_113.yaml` with the new image name
