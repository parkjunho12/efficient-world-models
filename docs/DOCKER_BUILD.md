# 🚀 Quick Docker Build Guide

## 문제 해결: requirements.txt not found

### 원인
Docker 빌드 시 `requirements.txt` 파일을 찾을 수 없는 경우는 보통 다음과 같은 이유 때문입니다:
1. `.dockerignore`에 의해 제외됨
2. 빌드 컨텍스트가 잘못 설정됨
3. 파일 경로 문제

### 해결 방법

#### 방법 1: 개선된 Dockerfile 사용 (권장)

새로운 Dockerfile은 `requirements.txt` 없이도 작동합니다:

```bash
# 빌드
docker build -t world-model:latest .

# 실행
docker run --gpus all -it world-model:latest
```

#### 방법 2: .dockerignore 확인

`.dockerignore` 파일에서 `requirements.txt`가 제외되지 않았는지 확인:

```bash
# .dockerignore 확인
cat .dockerignore | grep requirements.txt

# 있으면 제거
```

#### 방법 3: 빌드 컨텍스트 확인

올바른 디렉토리에서 빌드하는지 확인:

```bash
# 프로젝트 루트에서 실행
cd /path/to/world-model
ls -la  # setup.py, Dockerfile, src/ 등이 보여야 함
docker build -t world-model:latest .
```

---

## 빠른 빌드 명령어

### 1. Production 이미지 빌드

```bash
# 기본 빌드
docker build -t world-model:latest .

# 캐시 없이 빌드 (clean build)
docker build --no-cache -t world-model:latest .

# BuildKit 사용 (빠른 빌드)
DOCKER_BUILDKIT=1 docker build -t world-model:latest .
```

### 2. Development 이미지 빌드

```bash
docker build --target development -t world-model:dev .
```

### 3. 특정 GPU 아키텍처용 빌드

```bash
# CUDA 11.8 (기본)
docker build -t world-model:latest .

# CUDA 12.1
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 \
  -t world-model:cuda12 .
```

---

## Docker 없이 설치 (로컬 개발)

Docker를 사용하지 않는 경우:

```bash
# 1. Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. PyTorch 설치
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 나머지 패키지 설치
pip install numpy pillow opencv-python pandas scikit-learn \
    scipy matplotlib seaborn imageio pyyaml tensorboard \
    wandb tqdm h5py jupyter ipython

# 4. 프로젝트 설치
pip install -e .
```

---

## 빌드 검증

### 1. GPU 확인

```bash
docker run --gpus all world-model:latest \
  python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

**예상 출력:**
```
CUDA: True, Devices: 1
```

### 2. 패키지 확인

```bash
docker run world-model:latest python3 -c "
import torch
import torchvision
import numpy as np
import cv2
print('✓ All packages imported successfully')
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
"
```

### 3. 프로젝트 모듈 확인

```bash
docker run world-model:latest python3 -c "
from models.world_model import build_world_model
from training.losses import WorldModelLoss
from data.datasets.nuscenes import NuScenesDataset
print('✓ All project modules imported successfully')
"
```

---

## 일반적인 빌드 오류 해결

### 오류 1: "CUDA not available"

**해결:**
```bash
# NVIDIA Docker 런타임 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 테스트
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 오류 2: "No space left on device"

**해결:**
```bash
# Docker 정리
docker system prune -a --volumes

# 사용하지 않는 이미지 삭제
docker image prune -a

# 빌드 캐시 정리
docker builder prune
```

### 오류 3: "Cannot connect to Docker daemon"

**해결:**
```bash
# Docker 서비스 시작
sudo systemctl start docker

# Docker 상태 확인
sudo systemctl status docker

# 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER
newgrp docker
```

### 오류 4: 빌드가 매우 느림

**해결:**
```bash
# BuildKit 활성화 (병렬 빌드)
export DOCKER_BUILDKIT=1
docker build -t world-model:latest .

# 또는 docker-compose에서
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose build
```

---

## 최소 요구사항

### 하드웨어
- **CPU**: 4+ cores
- **RAM**: 16GB+ (권장: 32GB)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 50GB+ free space

### 소프트웨어
- **Docker**: 20.10+
- **NVIDIA Driver**: 525+ (CUDA 11.8 지원)
- **docker-compose**: 1.29+ (선택)

---

## 이미지 크기 최적화

### 현재 이미지 크기 확인

```bash
docker images world-model
```

### 최적화 팁

1. **멀티스테이지 빌드 사용** (이미 적용됨)
2. **불필요한 파일 제외** (.dockerignore 활용)
3. **레이어 최소화**:

```dockerfile
# ❌ 나쁜 예 (3 layers)
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get clean

# ✅ 좋은 예 (1 layer)
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

---

## 빠른 테스트

빌드 후 프로젝트가 제대로 작동하는지 빠르게 테스트:

```bash
# 1. 컨테이너 시작
docker run -it --gpus all \
  -v $(pwd)/data:/workspace/data \
  world-model:latest bash

# 2. 컨테이너 내부에서
python -c "
from models.world_model import build_world_model
import torch

model = build_world_model({
    'latent_dim': 256,
    'action_dim': 4,
    'hidden_dim': 512
})

# 테스트 입력
images = torch.randn(2, 10, 3, 256, 256)
actions = torch.randn(2, 9, 4)

# Forward pass
output = model(images, actions)
print('✓ Model works!')
print(f'Output shape: {output[\"reconstructed\"].shape}')
"
```

---

## 추가 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [NVIDIA Docker 문서](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker 이미지](https://hub.docker.com/r/pytorch/pytorch)
- [CUDA 호환성](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)

---

**문제가 계속되면 이슈를 남겨주세요!** 🐛