#!/bin/bash
#SBATCH --job-name=world-model
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out


set -euo pipefail

module purge

# CUDA 모듈 이름은 CUDA/xx.yy.zz 형태
module load CUDA/12.1.1

# miniconda 모듈은 없고 Anaconda3 모듈이 있음
module load Anaconda3/2024.02-1

echo "=== Python ==="
which python
python --version

# (선택) 가상환경: 매번 새로 만들지 말고 한 번 만들고 재사용
ENV_DIR="$HOME/.conda/envs/ewm"
if [ ! -d "$ENV_DIR" ]; then
  conda create -y -p "$ENV_DIR" python=3.11
fi
source activate "$ENV_DIR"

python --version
nvidia-smi

# pip 기본 세팅
python -m pip install -U pip setuptools wheel

conda install -y -c conda-forge numpy=1.26.4 scipy pandas scikit-learn opencv

python -m pip install --no-cache-dir \
  torch==2.1.0 torchvision==0.16.0 \
  --extra-index-url https://download.pytorch.org/whl/cu121


# requirements 설치
python -m pip install --no-cache-dir -r requirements.txt

# 실행
python scripts/train.py --config configs/training/base.yaml