# 🎓 Training Guide

Complete guide to training the World Model, from basic usage to advanced techniques.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Training Configuration](#training-configuration)
3. [Training Process](#training-process)
4. [Monitoring & Debugging](#monitoring--debugging)
5. [Advanced Techniques](#advanced-techniques)
6. [Troubleshooting](#troubleshooting)

---

## ⚡ Quick Start

### Basic Training

```bash
# 1. Prepare data (if not done already)
python scripts/prepare_dataset.py \
    --dataset nuscenes \
    --data-root data/nuscenes

# 2. Start training
python scripts/train.py \
    --config configs/training/base.yaml \
    --data-root data/processed/nuscenes

# 3. Monitor with TensorBoard
tensorboard --logdir runs
```

### Training with Weights & Biases

```bash
# Set API key
export WANDB_API_KEY=your_api_key_here

# Train (automatically logs to W&B)
python scripts/train.py \
    --config configs/training/base.yaml \
    --use-wandb
```

---

## ⚙️ Training Configuration

### Configuration Files

```
configs/
├── model/
│   └── base.yaml          # Model architecture
├── training/
│   ├── base.yaml          # Training hyperparameters
│   └── distributed.yaml   # Multi-GPU settings
└── data/
    ├── nuscenes.yaml      # nuScenes config
    └── carla.yaml         # CARLA config
```

### Base Configuration

**`configs/training/base.yaml`**:

```yaml
# Training
training:
  batch_size: 16
  num_epochs: 100
  use_amp: true              # Mixed precision
  gradient_accumulation: 4   # Effective batch: 64
  max_grad_norm: 1.0         # Gradient clipping

# Optimizer
optimizer:
  type: adamw
  lr: 1e-4
  encoder_lr: 1e-5          # Lower LR for encoder
  dynamics_lr: 1e-4
  decoder_lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Scheduler
scheduler:
  type: cosine
  warmup_steps: 1000
  min_lr: 1e-6

# Loss
loss:
  recon_weight: 1.0
  pred_weight: 1.0
  perceptual_weight: 0.1
  latent_reg_weight: 0.01
  temporal_weight: 0.1

# Data
data:
  num_workers: 4
  pin_memory: true
  sequence_length: 10
  frame_skip: 1

# Logging
logging:
  log_interval: 100          # Log every N steps
  eval_interval: 1000        # Evaluate every N steps
  save_interval: 5000        # Save checkpoint every N steps
  num_samples: 8             # Samples for visualization
```

### Model Configuration

**`configs/model/base.yaml`**:

```yaml
model:
  latent_dim: 256
  action_dim: 4
  hidden_dim: 512
  
  encoder:
    stages: [64, 128, 256, 512]
    num_heads: 4
  
  dynamics:
    num_layers: 4
    dropout: 0.1
  
  decoder:
    stages: [512, 256, 128, 64]
    init_size: 16
```

---

## 🏋️ Training Process

### Stage 1: Initialization (Epoch 0-10)

**Goal**: Learn basic reconstruction

**Characteristics**:
- High reconstruction loss (~0.5)
- Random predictions
- Unstable metrics

**What to watch**:
```
✓ Loss decreasing
✓ No NaN/Inf values
✓ GPU utilization > 80%
```

**Typical output**:
```
Epoch 1/100 [===>    ] Step 100/1000
  train/loss: 0.523
  train/psnr: 18.2 dB
  GPU: 3.2GB / 11GB
  Time: 2.3s/batch
```

### Stage 2: Learning (Epoch 10-50)

**Goal**: Improve reconstruction & start prediction

**Characteristics**:
- Steady loss decrease
- PSNR: 18 → 25 dB
- Predictions become meaningful

**What to watch**:
```
✓ PSNR increasing
✓ Perceptual loss decreasing
✓ Predictions look similar to targets
```

**Typical output**:
```
Epoch 25/100 [=====>  ] Step 5000/10000
  train/loss: 0.234
  train/psnr: 24.5 dB
  train/ssim: 0.82
  val/psnr: 24.1 dB
  Learning rate: 8.5e-5
```

### Stage 3: Refinement (Epoch 50-100)

**Goal**: Polish details and stabilize

**Characteristics**:
- Slow improvement
- PSNR: 25 → 28 dB
- High-quality predictions

**What to watch**:
```
✓ Validation metrics
✓ Overfitting signs
✓ Sample quality
```

**Typical output**:
```
Epoch 80/100 [=======>] Step 8000/10000
  train/loss: 0.156
  train/psnr: 27.8 dB
  train/ssim: 0.89
  val/psnr: 27.3 dB  ← Close to train!
  Best val: 27.5 dB (epoch 75)
```

### Expected Timeline

| Phase | Epochs | Time (RTX 3090) | PSNR |
|-------|--------|-----------------|------|
| **Initialization** | 0-10 | 1-2 hours | 15-20 dB |
| **Learning** | 10-50 | 4-6 hours | 20-26 dB |
| **Refinement** | 50-100 | 6-8 hours | 26-28 dB |
| **Total** | 100 | **10-16 hours** | **26.5-28.5 dB** |

---

## 📊 Monitoring & Debugging

### TensorBoard

**Start TensorBoard**:
```bash
tensorboard --logdir runs
# Open http://localhost:6006
```

**Available Metrics**:

1. **Scalars**:
   - `train/loss` - Total loss
   - `train/recon_loss` - Reconstruction loss
   - `train/pred_loss` - Prediction loss
   - `train/psnr` - PSNR metric
   - `train/ssim` - SSIM metric
   - `val/*` - Validation metrics
   - `lr` - Learning rate

2. **Images**:
   - `train/reconstruction` - GT vs Reconstructed
   - `train/prediction` - GT vs Predicted
   - `val/samples` - Validation samples

3. **Histograms**:
   - `weights/*` - Model weights
   - `gradients/*` - Gradient distributions

### Weights & Biases

**Features**:
- ✅ Cloud-based tracking
- ✅ Experiment comparison
- ✅ Hyperparameter sweeps
- ✅ Model versioning

**Setup**:
```bash
# Login
wandb login

# Train with W&B
python scripts/train.py \
    --config configs/training/base.yaml \
    --use-wandb \
    --wandb-project world-model \
    --wandb-name experiment-001
```

### What to Monitor

#### 1. Loss Curves

**Healthy Training**:
```
Loss
│
│ ╲
│  ╲_______________  ← Steady decrease, then plateau
│
└─────────────────── Epochs
```

**Problems**:
```
Loss
│    ╱╲╱╲
│   ╱  ╲ ╲          ← Oscillating (LR too high)
│  ╱    ╲  ╲
│ ╱      ╲  ╲
└───────────────── Epochs

Loss
│
│ ━━━━━━━━━━━━━    ← Plateau too early (LR too low)
│
└───────────────── Epochs
```

#### 2. PSNR/SSIM

**Target Ranges**:
- PSNR: 26.5-28.5 dB (nuScenes mini)
- SSIM: 0.85-0.90

**Warning Signs**:
- ⚠️ PSNR < 25 dB after 50 epochs
- ⚠️ Train/Val gap > 2 dB (overfitting)
- ⚠️ Metrics not improving for 10+ epochs

#### 3. Learning Rate

**Warmup Phase** (0-1000 steps):
```
LR
│        ___________
│       ╱
│      ╱
│     ╱
│____╱
└──────────────── Steps
   warmup=1000
```

**Cosine Annealing** (after warmup):
```
LR
│╲
│ ╲
│  ╲___
│      ╲___
│          ╲___
└─────────────── Steps
```

#### 4. GPU Utilization

**Check GPU usage**:
```bash
watch -n 1 nvidia-smi
```

**Target**: 85-95% GPU utilization

**Low utilization (<70%)?**
- Increase `num_workers`
- Increase `batch_size`
- Enable `pin_memory`
- Use faster storage (SSD)

---

## 🚀 Advanced Techniques

### 1. Mixed Precision Training

**Benefits**:
- 50% memory reduction
- 40% speed improvement
- Minimal accuracy loss

**Enable**:
```yaml
# configs/training/base.yaml
training:
  use_amp: true
```

**Code**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, actions in dataloader:
    optimizer.zero_grad()
    
    # Forward in FP16
    with autocast():
        outputs = model(images, actions)
        loss = criterion(outputs, targets)
    
    # Backward with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. Gradient Accumulation

**Benefits**:
- Larger effective batch size
- Fits in smaller GPU memory
- Better gradient estimates

**Example**:
```python
# Effective batch: 16 × 4 = 64
accumulation_steps = 4

for i, (images, actions) in enumerate(dataloader):
    with autocast():
        loss = model(images, actions)
        loss = loss / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 3. Distributed Training

**Single Machine, Multiple GPUs**:

```bash
# DataParallel (simple but slower)
python scripts/train.py --use-dp

# DistributedDataParallel (recommended)
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/training/distributed.yaml
```

**Configuration**:
```yaml
# configs/training/distributed.yaml
training:
  distributed: true
  backend: nccl
  batch_size: 16  # Per GPU
  gradient_accumulation: 1
```

**Expected Speedup**:
- 2 GPUs: 1.85×
- 4 GPUs: 3.46×
- 8 GPUs: 6.46×

### 4. Gradient Checkpointing

**Benefits**:
- 50% memory reduction
- ~20% slower training
- Allows larger models/batches

**Enable**:
```python
model = build_world_model(config)
model.enable_gradient_checkpointing()
```

### 5. Learning Rate Finder

**Find optimal LR**:
```bash
python scripts/find_lr.py \
    --config configs/training/base.yaml \
    --min-lr 1e-6 \
    --max-lr 1e-2
```

**Interpretation**:
```
Loss vs LR
│
│       ╱
│      ╱
│     ╱
│    ╱
│___╱
└─────────────── LR (log scale)
1e-6    1e-4    1e-2
          ↑
     Use this!
```

### 6. Curriculum Learning

**Strategy**: Start with easier examples

```python
# Sort by difficulty
sequences = sorted(sequences, key=lambda x: x['difficulty'])

# Training phases
epochs_per_phase = 20
for phase in range(3):
    start_idx = phase * len(sequences) // 3
    end_idx = (phase + 1) * len(sequences) // 3
    phase_sequences = sequences[start_idx:end_idx]
    
    # Train on this phase
    train(phase_sequences, epochs=epochs_per_phase)
```

---

## 🐛 Troubleshooting

### Problem 1: Loss Not Decreasing

**Symptoms**:
- Loss stays constant
- PSNR < 20 dB after 10 epochs

**Solutions**:

1. **Check learning rate**:
```python
# Try higher LR
optimizer.lr = 5e-4  # instead of 1e-4
```

2. **Check data**:
```python
# Visualize batch
batch = next(iter(dataloader))
show_batch(batch['images'])
```

3. **Check model output**:
```python
# Forward pass
outputs = model(images, actions)
print(outputs['reconstructed'].min(), outputs['reconstructed'].max())
# Should be in [-1, 1] range
```

### Problem 2: NaN Loss

**Symptoms**:
- Loss suddenly becomes NaN
- Gradients explode

**Solutions**:

1. **Lower learning rate**:
```yaml
optimizer:
  lr: 5e-5  # instead of 1e-4
```

2. **Gradient clipping**:
```yaml
training:
  max_grad_norm: 1.0  # Add if missing
```

3. **Check for bad data**:
```python
# Find NaN in dataset
for i, batch in enumerate(dataloader):
    if torch.isnan(batch['images']).any():
        print(f"NaN found in batch {i}")
```

### Problem 3: Overfitting

**Symptoms**:
- Train PSNR: 28 dB, Val PSNR: 24 dB
- Large train/val gap

**Solutions**:

1. **Increase regularization**:
```yaml
loss:
  latent_reg_weight: 0.05  # increase from 0.01
```

2. **Add dropout**:
```yaml
model:
  dynamics:
    dropout: 0.2  # increase from 0.1
```

3. **More data augmentation**:
```python
transform = DrivingAugmentation(
    brightness=0.3,  # increase
    contrast=0.3,
    crop_scale=(0.7, 1.0)  # more aggressive
)
```

### Problem 4: Slow Training

**Symptoms**:
- < 2 batches/second
- Low GPU utilization (< 70%)

**Solutions**:

1. **Increase num_workers**:
```yaml
data:
  num_workers: 8  # increase from 4
```

2. **Enable pin_memory**:
```yaml
data:
  pin_memory: true
```

3. **Use faster storage**:
```bash
# Move data to SSD
rsync -av --progress /hdd/data/ /ssd/data/
```

4. **Reduce logging frequency**:
```yaml
logging:
  log_interval: 500  # instead of 100
```

### Problem 5: Out of Memory

**Symptoms**:
- CUDA out of memory error
- Training crashes

**Solutions**:

1. **Reduce batch size**:
```yaml
training:
  batch_size: 8  # instead of 16
```

2. **Enable gradient checkpointing**:
```yaml
training:
  use_gradient_checkpointing: true
```

3. **Use gradient accumulation**:
```yaml
training:
  batch_size: 4
  gradient_accumulation: 4  # Effective: 16
```

4. **Mixed precision**:
```yaml
training:
  use_amp: true
```

---

## 📈 Optimization Checklist

### Before Training

- [ ] Data preprocessed and validated
- [ ] Config file reviewed
- [ ] GPU drivers updated
- [ ] Disk space sufficient (>50GB)
- [ ] TensorBoard/W&B ready

### During Training

- [ ] Monitor loss curves
- [ ] Check sample predictions
- [ ] Watch GPU utilization
- [ ] Verify checkpoints saving
- [ ] Compare train/val metrics

### After Training

- [ ] Evaluate on test set
- [ ] Run benchmark script
- [ ] Generate visualizations
- [ ] Save best checkpoint
- [ ] Document hyperparameters

---

## 🎯 Training Recipes

### Recipe 1: Quick Prototype (2 hours)

```yaml
# Fast training for debugging
training:
  batch_size: 32
  num_epochs: 20
  use_amp: true

data:
  sequence_length: 5  # Shorter sequences
  num_workers: 8

model:
  latent_dim: 128  # Smaller model
  hidden_dim: 256
```

### Recipe 2: Best Quality (24 hours)

```yaml
# Maximum quality
training:
  batch_size: 16
  num_epochs: 200
  use_amp: true
  gradient_accumulation: 4

model:
  latent_dim: 512  # Larger model
  hidden_dim: 1024
  
loss:
  perceptual_weight: 0.2  # More perceptual
```

### Recipe 3: Memory Efficient (8GB GPU)

```yaml
# Fits on smaller GPUs
training:
  batch_size: 4
  gradient_accumulation: 4
  use_amp: true
  use_gradient_checkpointing: true

model:
  latent_dim: 128
  hidden_dim: 256
```

---

## 📚 Additional Resources

- [Architecture Guide](ARCHITECTURE.md) - Model details
- [Dataset Guide](DATASETS.md) - Data preparation
- [Docker Guide](DOCKER_GUIDE.md) - Container training
- [Utilities Guide](UTILITIES_GUIDE.md) - Helper functions

---

**Ready to train?** Start with the base config and adjust as needed! 🎓