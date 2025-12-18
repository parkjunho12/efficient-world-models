# 📊 Dataset Guide

Complete guide to setting up datasets for world model training.

## 📋 Table of Contents

1. [Quick Recommendation](#quick-recommendation)
2. [nuScenes Dataset](#1-nuscenes-recommended)
3. [CARLA Simulator](#2-carla-simulator)
4. [Waymo Open Dataset](#3-waymo-open-dataset)
5. [Custom Datasets](#4-custom-datasets)
6. [Data Format](#data-format)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Recommendation

### For Amazon/Wayve Interviews: **nuScenes mini (11GB)** ⭐

**Why?**
- ✅ **Industry-standard** - Everyone recognizes it
- ✅ **Fast setup** - 1-2 days total (download + train)
- ✅ **Credible results** - PSNR 26.5-28.5 dB expected
- ✅ **Interview-friendly** - Easy to explain and compare

### For Quick Experiments: **CARLA** ⚡

**Why?**
- ✅ **Zero download** - Generate data on the fly
- ✅ **Perfect for debugging** - Full control over scenarios
- ✅ **Fast iteration** - 2-3 hours to generate + train

---

## 1. nuScenes (⭐ Recommended)

### Overview

- **Size**: 11GB (mini) or 350GB (full)
- **Scenes**: 10 (mini) or 1,000 (full)
- **Cameras**: 6 (front, front-left, front-right, back, back-left, back-right)
- **Sensors**: Camera, LiDAR, RADAR, GPS/IMU
- **Location**: Boston and Singapore
- **Website**: https://www.nuscenes.org/

### Setup Instructions

#### Step 1: Download

**Option A: Automatic Download (Recommended)**

```bash
# Download mini split (11GB, ~1-2 hours)
python scripts/download_nuscenes.py \
    --split mini \
    --output data/nuscenes

# Download full split (350GB, ~1-2 days)
python scripts/download_nuscenes.py \
    --split trainval \
    --output data/nuscenes
```

**Option B: Manual Download**

1. Register at https://www.nuscenes.org/
2. Download from: https://www.nuscenes.org/download
3. Extract to `data/nuscenes/`

```bash
# Expected structure
data/nuscenes/
├── v1.0-mini/
│   ├── samples/        # Camera images
│   ├── sweeps/         # LiDAR sweeps
│   └── *.json          # Metadata
└── ...
```

#### Step 2: Preprocess

```bash
# Preprocess for training (10-20 minutes for mini)
python scripts/prepare_dataset.py \
    --dataset nuscenes \
    --data-root data/nuscenes \
    --output data/processed/nuscenes \
    --sequence-length 10 \
    --image-size 256 256
```

**Output format:**
```
data/processed/nuscenes/
├── train/
│   ├── scene_0001/
│   │   ├── images/          # 000.jpg, 001.jpg, ...
│   │   ├── actions.npy      # (T-1, 4) vehicle controls
│   │   └── metadata.json    # Scene metadata
│   └── scene_0002/
│       └── ...
└── val/
    └── ...
```

#### Step 3: Train

```bash
# Training config
python scripts/train.py \
    --config configs/training/base.yaml \
    --data-root data/processed/nuscenes \
    --dataset nuscenes

# Monitor progress
tensorboard --logdir runs
```

### Expected Results

| Metric | Mini (10 scenes) | Full (1000 scenes) |
|--------|------------------|-------------------|
| **PSNR** | 26.5-28.5 dB | 28.5-30.5 dB |
| **SSIM** | 0.85-0.90 | 0.90-0.93 |
| **Training Time** | 6-8 hours | 2-3 days |
| **GPU Memory** | ~10GB | ~12GB |
| **Dataset Size** | 11GB | 350GB |

**Baseline (RTX 3090):**
- Latency: ~42ms per sequence
- Throughput: 23.8 FPS
- Model size: 28M parameters

### Tips & Tricks

**1. Use CAM_FRONT only** (default)
```python
# In configs/data/nuscenes.yaml
camera: CAM_FRONT  # Simplest setup
```

**2. Increase sequence length** (more context)
```python
sequence_length: 15  # instead of 10
```

**3. Frame skip** (faster training)
```python
frame_skip: 2  # Use every 2nd frame
```

**4. Cache images** (faster loading, more RAM)
```python
# In dataset initialization
cache_images: true  # Only for mini!
```

---

## 2. CARLA Simulator

### Overview

- **Size**: 0GB (generate on-the-fly!)
- **Scenarios**: Unlimited
- **Customization**: Full control
- **Website**: https://carla.org/

### Setup Instructions

#### Step 1: Install CARLA

**Option A: Pre-built Binary (Recommended)**

```bash
# Download CARLA 0.9.15 (~10GB)
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz

# Extract
tar -xzf CARLA_0.9.15.tar.gz

# Run server
cd CARLA_0.9.15
./CarlaUE4.sh
```

**Option B: Docker**

```bash
docker run -d \
    -p 2000-2002:2000-2002 \
    --runtime=nvidia \
    carlasim/carla:0.9.15 \
    /bin/bash CarlaUE4.sh -RenderOffScreen
```

#### Step 2: Generate Data

```bash
# Generate training data (2-3 hours for 500 episodes)
python scripts/collect_carla_data.py \
    --num-episodes 500 \
    --output data/carla \
    --towns Town01,Town02,Town03 \
    --weather ClearNoon,CloudyNoon,WetNoon

# Quick test (10 episodes, 5 minutes)
python scripts/collect_carla_data.py \
    --num-episodes 10 \
    --output data/carla_test
```

**Data collection settings:**
- Episode length: ~200 frames (~10 seconds at 20 FPS)
- Image size: 256x256
- Actions: [steering, throttle, brake, gear]
- Autopilot mode: Enabled

#### Step 3: Preprocess (Optional)

```bash
# CARLA data is already in the right format!
# But you can preprocess for consistency
python scripts/prepare_dataset.py \
    --dataset carla \
    --data-root data/carla
```

#### Step 4: Train

```bash
python scripts/train.py \
    --config configs/training/base.yaml \
    --data-root data/carla \
    --dataset carla
```

### Expected Results

| Metric | 500 Episodes | 2000 Episodes |
|--------|--------------|---------------|
| **PSNR** | 24-26 dB | 26-28 dB |
| **SSIM** | 0.80-0.85 | 0.85-0.90 |
| **Training Time** | 4-6 hours | 12-16 hours |
| **Data Generation** | 2-3 hours | 8-10 hours |

### Advanced: Custom Scenarios

```python
# In scripts/collect_carla_data.py

# Custom weather
WEATHER_PRESETS = [
    'ClearNoon',
    'CloudyNoon', 
    'WetNoon',
    'HardRainNoon',
    'ClearSunset'
]

# Custom towns
TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']

# Custom sensors
# Add multiple cameras, LiDAR, etc.
```

### Pros & Cons

**Pros:**
- ✅ No download needed
- ✅ Unlimited data
- ✅ Perfect for debugging
- ✅ Full control over scenarios

**Cons:**
- ❌ Synthetic (not real-world)
- ❌ Less credible for interviews
- ❌ Requires CARLA installation

---

## 3. Waymo Open Dataset

### Overview

- **Size**: ~1TB (full)
- **Scenes**: 1,000+ hours of driving
- **Sensors**: 5 cameras, 5 LiDAR, GPS/IMU
- **Location**: Phoenix, San Francisco, etc.
- **Website**: https://waymo.com/open/

### Setup Instructions

⚠️ **Warning**: Very large dataset. Only recommended for serious research.

#### Step 1: Download

```bash
# Download (WARNING: ~1TB!)
python scripts/download_waymo.py \
    --output data/waymo \
    --split training

# This will take several days!
```

#### Step 2: Preprocess

```bash
python scripts/prepare_dataset.py \
    --dataset waymo \
    --data-root data/waymo \
    --output data/processed/waymo
```

#### Step 3: Train

```bash
python scripts/train.py \
    --config configs/training/base.yaml \
    --data-root data/processed/waymo \
    --dataset waymo
```

### Expected Results

| Metric | Waymo |
|--------|-------|
| **PSNR** | 29-31 dB |
| **SSIM** | 0.91-0.94 |
| **Training Time** | 5-7 days |
| **Dataset Size** | ~1TB |

### Tips

- Use a subset (e.g., 100 scenes) for testing
- Consider using cloud storage (AWS S3, GCS)
- Preprocess on a powerful machine

---

## 4. Custom Datasets

### Format Requirements

To add your own dataset, follow this structure:

```
data/my_dataset/
├── train/
│   ├── sequence_001/
│   │   ├── images/
│   │   │   ├── 000.jpg
│   │   │   ├── 001.jpg
│   │   │   └── ...
│   │   ├── actions.npy      # Shape: (T-1, action_dim)
│   │   └── metadata.json    # Optional
│   └── sequence_002/
│       └── ...
└── val/
    └── ...
```

### Implementation

Create a custom dataset class:

```python
# src/data/datasets/my_dataset.py

from .base import BaseDataset

class MyDataset(BaseDataset):
    """Custom dataset loader"""
    
    def _load_sequences(self):
        """Load sequence metadata"""
        sequences = []
        # Your logic here
        return sequences
    
    def __getitem__(self, idx):
        """Load a sequence"""
        # Your logic here
        return {
            'images': images,  # (T, 3, H, W)
            'actions': actions,  # (T-1, A)
            'scene_id': scene_id
        }
```

Register in config:

```yaml
# configs/data/my_dataset.yaml
dataset:
  name: my_dataset
  data_root: data/my_dataset
  sequence_length: 10
```

---

## 📊 Data Format

### Directory Structure

```
data/processed/{dataset}/
├── train/
│   ├── scene_0001/
│   │   ├── images/          # Sequential frames
│   │   │   ├── 000.jpg
│   │   │   ├── 001.jpg
│   │   │   └── ...
│   │   ├── actions.npy      # (T-1, action_dim) - controls between frames
│   │   └── metadata.json    # Scene info (optional)
│   ├── scene_0002/
│   └── ...
└── val/
    └── ...
```

### File Formats

#### 1. Images (`images/`)
- **Format**: JPEG or PNG
- **Size**: 256x256 (default) or 512x512
- **Naming**: Zero-padded numbers (000.jpg, 001.jpg, ...)
- **Color space**: RGB

#### 2. Actions (`actions.npy`)
- **Format**: NumPy array
- **Shape**: `(T-1, action_dim)` where T is sequence length
- **dtype**: float32
- **Range**: [-1, 1] or [0, 1] depending on normalization

**Action dimensions** (example for driving):
```python
actions = np.array([
    [steering, throttle, brake, gear],  # Frame 0 -> 1
    [steering, throttle, brake, gear],  # Frame 1 -> 2
    ...
])
```

#### 3. Metadata (`metadata.json`)
- **Format**: JSON
- **Content**: Scene information

```json
{
  "scene_id": "scene_0001",
  "location": "Boston",
  "weather": "sunny",
  "time_of_day": "noon",
  "num_frames": 200,
  "fps": 20
}
```

---

## 🐛 Troubleshooting

### Issue 1: Download Failed

**nuScenes:**
```bash
# Resume download
python scripts/download_nuscenes.py --split mini --resume

# Or download manually from website
```

**CARLA:**
```bash
# Check CARLA is running
ps aux | grep Carla

# Restart server
./CarlaUE4.sh -RenderOffScreen
```

### Issue 2: Preprocessing Slow

```bash
# Use more workers
python scripts/prepare_dataset.py \
    --dataset nuscenes \
    --num-workers 8  # Increase from default 4

# Or process in chunks
python scripts/prepare_dataset.py --start-idx 0 --end-idx 100
python scripts/prepare_dataset.py --start-idx 100 --end-idx 200
```

### Issue 3: Out of Disk Space

```bash
# Check space
df -h

# Clean up
rm -rf data/nuscenes/samples/  # After preprocessing
rm -rf data/carla_raw/         # Keep only processed

# Use external drive
ln -s /mnt/external/data ./data
```

### Issue 4: Data Loading Slow

```bash
# Enable caching (small datasets only)
cache_images: true

# Increase num_workers
num_workers: 8

# Use SSD instead of HDD
mv data/ /ssd/data/
ln -s /ssd/data ./data
```

---

## 📈 Dataset Comparison

| Dataset | Size | Scenes | Real/Sim | Difficulty | Interview Score |
|---------|------|--------|----------|------------|----------------|
| **nuScenes mini** | 11GB | 10 | Real | Easy | ⭐⭐⭐⭐⭐ |
| **CARLA** | 0GB* | ∞ | Sim | Easy | ⭐⭐⭐ |
| **nuScenes full** | 350GB | 1000 | Real | Medium | ⭐⭐⭐⭐⭐ |
| **Waymo** | 1TB | 1000+ | Real | Hard | ⭐⭐⭐⭐ |

*Generate on-the-fly

---

## 🎯 Recommendations by Goal

### For Interviews (Amazon, Wayve)
1. **nuScenes mini** (Week 1-2)
2. nuScenes full (Week 3+, optional)

### For Quick Prototyping
1. **CARLA** (Day 1)
2. nuScenes mini (Week 2)

### For Research Papers
1. nuScenes full
2. Waymo (for comparison)
3. CARLA (for ablations)

### For Production Systems
1. Custom dataset (your fleet)
2. Waymo (for benchmarking)

---

**Ready to download data?** Start with nuScenes mini or CARLA! 📊