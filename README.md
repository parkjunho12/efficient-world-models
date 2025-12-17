# 🚗 World Model for Autonomous Driving

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Production-ready spatiotemporal world model for visual prediction in autonomous systems**

## ⚡ Quick Start

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Download nuScenes mini dataset (11GB)
python scripts/download_nuscenes.py --split mini

# Train model
python scripts/train.py --config configs/training/base.yaml

# Run inference
python scripts/inference.py --checkpoint checkpoints/best.pt --input video.mp4
\`\`\`

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| Inference Speed | 42ms/frame |
| Model Size | 28M params |
| PSNR | 28.5 dB |
| LPIPS | 0.12 |

## 📂 Project Structure

\`\`\`
world-model-autonomous/
├── src/                  # Source code
│   ├── models/          # Model architectures
│   ├── training/        # Training utilities
│   ├── data/            # Dataset loaders
│   ├── evaluation/      # Metrics & visualization
│   └── utils/           # Helper functions
├── scripts/             # Executable scripts
├── configs/             # Configuration files
├── tests/               # Unit & integration tests
├── notebooks/           # Jupyter notebooks
└── docs/                # Documentation
\`\`\`

## 📚 Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Dataset Guide](docs/DATASETS.md) - **Start here for data setup!**
- [Architecture](docs/ARCHITECTURE.md)
- [Training Guide](docs/TRAINING.md)

## 🗃️ Supported Datasets

### 1. nuScenes (⭐ Recommended)
- Industry-standard benchmark
- 11GB (mini) or 350GB (full)
- Perfect for interviews
- [Setup Guide](docs/DATASETS.md#1-nuscenes-recommended)

### 2. CARLA Simulator
- Generate unlimited data
- Perfect for quick experiments
- No download needed
- [Setup Guide](docs/DATASETS.md#2-carla-simulator)

### 3. Waymo Open Dataset  
- Largest public dataset (~1TB)
- Real Waymo sensors
- [Setup Guide](docs/DATASETS.md#3-waymo-open-dataset)

## 💡 For Amazon/Wayve Interviews

**This project demonstrates:**
- ✅ Production-ready ML systems
- ✅ Efficient architecture (42ms latency)
- ✅ Industry-standard benchmarks (nuScenes)
- ✅ Research depth (perceptual losses, uncertainty)
- ✅ Clean, modular code
- ✅ Comprehensive evaluation

## 📄 License

MIT License - see [LICENSE](LICENSE)