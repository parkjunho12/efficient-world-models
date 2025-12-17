# World Model for Autonomous Driving - Dockerfile
# 
# Multi-stage build for optimized production deployment
#
# Usage:
#   docker build -t world-model:latest .
#   docker run --gpus all -v $(pwd)/data:/workspace/data world-model:latest

# ============================================================================
# Stage 1: Base Image with CUDA
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Dependencies
# ============================================================================
FROM base AS dependencies

ENV PYTHONUNBUFFERED=1
WORKDIR /tmp

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel \
 && python3 -m pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
      --index-url https://download.pytorch.org/whl/cu118 \
 && python3 -m pip install --no-cache-dir --use-deprecated=legacy-resolver -r /tmp/requirements.txt


# Install additional ML tools
RUN pip3 install \
    tensorboard \
    wandb \
    opencv-python \
    scikit-learn \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    jupyter \
    ipython

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

# Set working directory
WORKDIR /workspace

# Copy application code
COPY . /workspace/

# Install package in development mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p \
    /workspace/data \
    /workspace/checkpoints \
    /workspace/runs \
    /workspace/outputs \
    /workspace/logs

# Set permissions
RUN chmod -R 755 /workspace

# ============================================================================
# Stage 4: Production (Final)
# ============================================================================
FROM application AS production

# Set default command
CMD ["/bin/bash"]

# Expose ports
EXPOSE 6006 
EXPOSE 8888  

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print(torch.cuda.is_available())" || exit 1

# Add labels
LABEL maintainer="ghdlwnsgh25@gmail.com" \
      version="1.0" \
      description="World Model for Autonomous Driving"

# ============================================================================
# Development Stage (Optional)
# ============================================================================
FROM application AS development

# Install development tools
RUN pip3 install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit \
    ipdb

# Set development environment
ENV ENVIRONMENT=development

CMD ["/bin/bash"]