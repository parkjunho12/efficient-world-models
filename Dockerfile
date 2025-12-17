# syntax=docker/dockerfile:1.7

FROM python:3.10-slim

# -------- 기본 환경 설정 --------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# -------- 시스템 패키지 --------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------- Python 의존성 --------
# requirements.txt가 있으면 먼저 복사 (캐시 효율)
COPY requirements.txt ./requirements.txt
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# -------- 소스 코드 --------
COPY . .

# -------- 기본 실행 --------
# 필요에 따라 변경
CMD ["python", "-m", "app"]
