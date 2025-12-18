# 🐳 Docker Deployment Guide

## 📋 Project Structure

```
.
├── Dockerfile              # Multi-stage Docker image definition
├── docker-compose.yml      # Service orchestration
├── .dockerignore           # Docker build exclusions
├── requirements.txt        # Python dependencies
└── docs/DOCKER_GUIDE.md    # This document

```

---

## 🚀 Quick Start

### 1.Build Docker Images

```bash
# Production image
docker build -t world-model:latest .

# Development image
docker build --target development -t world-model:dev .

```

### 2. Run the Full Stack with Docker Compose

```bash
# Start all services
docker-compose up -d

# Start specific services only
docker-compose up -d world-model-train tensorboard

# View logs
docker-compose logs -f world-model-train

# Stop all services
docker-compose down

```

---

## 🎯 Individual Service Usage

### 1. Training

```bash
# Using Docker Compose
docker-compose up world-model-train

# Run directly
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/runs:/workspace/runs \
  world-model:latest \
  python scripts/train.py --config configs/training/base.yaml

```

### 2. TensorBoard

```bash
# Using Docker Compose
docker-compose up -d tensorboard

# Access in browser
# http://localhost:6006

# Run directly
docker run -d \
  -p 6006:6006 \
  -v $(pwd)/runs:/workspace/runs \
  world-model:latest \
  tensorboard --logdir=/workspace/runs --host=0.0.0.0

```

### 3. Jupyter Notebook

```bash
# Using Docker Compose
docker-compose up -d jupyter

# Check token
docker-compose logs jupyter

# Access in browser
# http://localhost:8888

# Run directly
docker run -d \
  --gpus all \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  world-model:latest \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

```

### 4. Evaluation

```bash
# Using Docker Compose
docker-compose up world-model-eval

# Run directly
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/outputs:/workspace/outputs \
  world-model:latest \
  python scripts/evaluate.py --checkpoint checkpoints/checkpoint_best.pt

```

### 5. Development Shell

```bash
# Using Docker Compose
docker-compose run --rm world-model-dev

# Run directly
docker run -it --gpus all \
  -v $(pwd):/workspace \
  world-model:dev \
  /bin/bash

```

---

## 📊 Dockerfile Architecture

### 멀티스테이지 빌드

```dockerfile
Stage 1: base           # CUDA + system dependencies
  ↓
Stage 2: dependencies   # Python packages
  ↓
Stage 3: application    # Application source code
  ↓
Stage 4: production     # Final production image (optimized)
  ↓
Stage 5: development    # Dev tools included (optional)

```

**Advantages:**
- Minimal final image size
- Optimized build cache usage
- Clear separation between development and production

### Included Core Packages

**Deep Learning:**
- PyTorch 2.1.0 (CUDA 11.8)
- TorchVision 0.16.0

**Visualization:**
- TensorBoard
- Weights & Biases
- Matplotlib, Seaborn

**Data Processing:**
- NumPy, Pandas, SciPy
- OpenCV, PIL

**Development (development image only):**
- Pytest, Black, Flake8
- Jupyter, IPython

---

## 🔧 Environment Variables

### Required Variables

```bash
# GPU configuration
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0,1

# W&B API key (optional)
WANDB_API_KEY=your_api_key_here

# Environment mode
ENVIRONMENT=production  # or development

```

### Configuration Methods

**1. In docker-compose.yml**
```yaml
environment:
  - WANDB_API_KEY=${WANDB_API_KEY}
  - CUDA_VISIBLE_DEVICES=0
```

**2. Using a .env file**
```bash
# .env
WANDB_API_KEY=your_key_here
CUDA_VISIBLE_DEVICES=0,1
```

**3. At runtime**
```bash
docker run -e WANDB_API_KEY=your_key world-model:latest
```

---

## 📂 Volume Mounts

### Recommended Mount Points

```yaml
volumes:
  - ./data:/workspace/data                    # Datasets
  - ./checkpoints:/workspace/checkpoints      # Model checkpoints
  - ./runs:/workspace/runs                    # TensorBoard logs
  - ./outputs:/workspace/outputs              # Result files
  - ./logs:/workspace/logs                    # Application logs

```

### Notes

**❌ Do NOT mount:**
- `__pycache__/` (auto)
- Python virtual environments (`venv/`, `env/`)
- Build artifacts

**✅ Recommended to mount:**
- Training data
- Configuration files
- Checkpoints
- Logs and outputs

---

## 🎛️ GPU Configuration

### Enable GPU

**Docker:**
```bash
docker run --gpus all world-model:latest
```

**Docker Compose:**
```yaml
runtime: nvidia
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1  # 사용할 GPU 수
          capabilities: [gpu]
```

### Select Specific GPUs

```bash
docker run --gpus '"device=0,1"' world-model:latest

# Or via environment variable
docker run -e CUDA_VISIBLE_DEVICES=0,1 world-model:latest
```

### GPU 메모리 제한

```yaml
deploy:
  resources:
    limits:
      memory: 16G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 🧪 Testing & Debugging

### 1. Check GPU Availability

```bash
docker run --gpus all world-model:latest \
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Access Container Shell

```bash
# Running container
docker exec -it world-model-train /bin/bash

# New container
docker run -it --rm world-model:latest /bin/bash

```

### 3. View Logs

```bash
# Real-time logs
docker-compose logs -f world-model-train

# Recent 100 lines
docker-compose logs --tail=100 world-model-train
```

### 4.Debug Mode

```bash
docker run -it --gpus all \
  -v $(pwd):/workspace \
  world-model:dev \
  python -m ipdb scripts/train.py
```

---

## 🚀 Production Deployment

### 1. Optimized Image Build

```bash
docker build \
  --build-arg CUDA_VERSION=11.8.0 \
  --build-arg PYTHON_VERSION=3.10 \
  -t world-model:prod .

```

### 2. Multi-GPU Training

```bash
docker run --gpus all \
  --shm-size=16g \
  -v $(pwd)/data:/workspace/data \
  world-model:latest \
  torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/training/distributed.yaml
```

### 3. Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 32G
    reservations:
      devices:
        - driver: nvidia
          count: 2
          capabilities: [gpu]
```

---

## 🔍 Troubleshooting
### Issue 1: GPU Not Detected

**Solution:**
```bash
# NVIDIA Docker runtime check
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Restart Docker
sudo systemctl restart docker
```

### Issue 2: Out of Memory

**Solution:**
```bash
# Increase Shared memory
docker run --shm-size=16g world-model:latest

# or docker-compose.yml
shm_size: '16gb'
```

### Issue 3: Permission Errors

**Solution:**
```bash
# Current user ID
docker run --user $(id -u):$(id -g) world-model:latest

# or modify volumes permission
sudo chown -R $(whoami):$(whoami) ./data ./checkpoints
```

### Issue 4: Slow Build

**Solution:**
```bash
# Activate BuildKit (Parellel build)
DOCKER_BUILDKIT=1 docker build -t world-model:latest .

# Using build cache
docker build --cache-from world-model:latest -t world-model:latest .
```

---

## 📈 Performance Optimisation

### 1. Build Optimisation

```dockerfile
# Optimise Layer Caching
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # 코드는 마지막에
```

### 2. Runtime Optimisation

```bash
# Fix CPU (NUMA Optimisation)
docker run --cpuset-cpus="0-7" world-model:latest

# I/O Priority
docker run --blkio-weight=500 world-model:latest
```

### 3. Network Optimisation

```yaml
# Using Host network (distributed train)
network_mode: host
```

---

## 📊 Monitoring

### Resource Usage

```bash
# Real-time statistics
docker stats world-model-train

# All containers
docker-compose stats
```

### GPU Monitoring

```bash
# In contatiner
docker exec world-model-train nvidia-smi

# Periodical checking
docker exec world-model-train watch -n 1 nvidia-smi
```

---
