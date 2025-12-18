# Getting Started

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/world-model-autonomous.git
cd world-model-autonomous

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Test

```python
import torch
from src.models.world_model import build_world_model

# Build model
config = {
    'latent_dim': 256,
    'action_dim': 4,
    'hidden_dim': 512,
    'num_layers': 4
}

model = build_world_model(config)

# Test forward pass
images = torch.randn(2, 5, 3, 256, 256)
actions = torch.randn(2, 4, 4)

outputs = model(images, actions)
print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
print(f"Predicted shape: {outputs['predicted'].shape}")
```

## Download Data

See [DATASETS.md](DATASETS.md) for detailed instructions.

**Quick start with nuScenes mini:**

```bash
python scripts/download_nuscenes.py --split mini --output data/nuscenes
python scripts/prepare_dataset.py --dataset nuscenes --data-root data/nuscenes
```

## Train Model

```bash
python scripts/train.py --config configs/training/base.yaml
```

## Next Steps

- Read [DATASETS.md](DATASETS.md) for dataset setup
- Read [TRAINING.md](TRAINING.md) for training tips
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for model details