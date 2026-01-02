# VoxCeleb Training Guide - Performance Optimized

This guide explains how to run the performance-optimized VoxCeleb speaker recognition training.

## Quick Start

### 1. Basic Training (Foreground)

Run training directly in your current terminal:

```bash
cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer

python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --save_path exps/exp_$(date +%Y%m%d_%H%M%S)_node119 \
  2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

⚠️ **Warning**: This will occupy your terminal. If you close it, training stops.

---

### 2. Background Training with tmux (Recommended)

Start training in a detached tmux session:

```bash
cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer

tmux new-session -d -s "train_voxceleb" \
"python3 trainSpeakerNet_performance_updated.py \
--config configs/experiment_01_performance_updated.yaml \
--save_path exps/exp_$(date +%Y%m%d_%H%M%S)_node119 \
2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

**Benefits**:
- ✅ Runs in background
- ✅ Survives terminal disconnection
- ✅ Can attach/detach anytime
- ✅ Keeps log files

---

### 3. Using the Convenience Script

The easiest way - use the provided script:

```bash
cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer
./run_performance_optimized.sh train
```

---

## Monitoring Training

### Attach to tmux Session

```bash
# List all sessions
tmux list-sessions

# Attach to your training session
tmux attach -t train_voxceleb
```

**Detach without stopping**: Press `Ctrl+B`, then press `D`

### View Log Files

```bash
# View live log (updates in real-time)
tail -f training_20251025_083738.log

# View last 50 lines
tail -50 training_20251025_083738.log

# Search log for specific text
grep "VEER" training_20251025_083738.log
```

### Monitor GPU Usage

```bash
# Check GPU utilization
nvidia-smi

# Watch GPU usage continuously (updates every 1 second)
watch -n 1 nvidia-smi
```

### Check Training Process

```bash
# Check if training is running
ps aux | grep trainSpeakerNet

# Check GPU processes
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

---

## Training Configuration

Default settings in `configs/experiment_01_performance_updated.yaml`:

```yaml
# Data Loading - OPTIMIZED
batch_size: 128              # Increased for better GPU utilization
nDataLoaderThread: 8         # More workers for faster data loading
prefetch_factor: 3           # Prefetch batches per worker
persistent_workers: true     # Keep workers alive between epochs

# Training
test_interval: 3             # Validate every 3 epochs
max_epoch: 500              # Maximum training epochs
optimizer: adam
lr: 0.001                   # Learning rate
lr_decay: 0.95              # Decay rate

# Validation
max_test_pairs: 10000       # Use 10K pairs (~75 seconds)
eval_batch_size: 64         # Batch size for evaluation

# Mixed Precision
mixedprec: true             # FP16 training enabled by default
```

---

## Command Line Options

### Override Config Settings

```bash
# Change batch size
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --batch_size 256

# Change validation frequency
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --test_interval 5

# Use more test pairs for validation
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --max_test_pairs 50000
```

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | - | Path to YAML config file |
| `--save_path` | `exps/exp1` | Where to save models |
| `--initial_model` | - | Pre-trained model to start from |
| `--batch_size` | 128 | Training batch size |
| `--test_interval` | 3 | Validate every N epochs |
| `--max_test_pairs` | 10000 | Number of test pairs (0=all) |
| `--eval_batch_size` | 64 | Evaluation batch size |
| `--lr` | 0.001 | Learning rate |
| `--max_epoch` | 500 | Maximum epochs |

---

## Example Commands

### Quick Test Training (Fast Validation)

```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --save_path exps/test_run \
  --max_test_pairs 1000 \
  --test_interval 1 \
  --max_epoch 10
```

### Production Training (Accurate Validation)

```bash
tmux new-session -d -s "prod_train" \
"python3 trainSpeakerNet_performance_updated.py \
--config configs/experiment_01_performance_updated.yaml \
--save_path exps/exp_$(date +%Y%m%d_%H%M%S)_production \
--max_test_pairs 50000 \
--test_interval 5 \
2>&1 | tee training_production_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

### Resume from Checkpoint

```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --save_path exps/resumed_training \
  --initial_model exps/exp_20251025_083738/model/model000000003.model
```

### Transfer Learning from Baseline

```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --save_path exps/transfer_learning \
  --initial_model /mnt/ricproject3/2025/data/baseline_v2_smproto.model
```

---

## Output Files

Training creates the following structure:

```
exps/exp_20251025_083738_node119/
├── model/                          # Saved models
│   ├── model000000003.model       # Model at epoch 3
│   ├── model000000006.model       # Model at epoch 6
│   └── best_model.model           # Best model (lowest EER)
├── result/
│   ├── scores.txt                 # Validation scores per epoch
│   ├── best_eer.txt              # Best EER achieved
│   └── best_threshold.txt        # Best decision threshold
└── logs/                          # TensorBoard logs
    └── events.out.tfevents.*
```

### View TensorBoard Logs

```bash
tensorboard --logdir exps/exp_20251025_083738_node119/logs --port 6006
```

Then open in browser: `http://localhost:6006`

---

## Troubleshooting

### Training Stops Unexpectedly

**Check if process is running**:
```bash
ps aux | grep trainSpeakerNet
```

**Check log file for errors**:
```bash
tail -100 training_20251025_083738.log | grep -i error
```

### Out of Memory (OOM) Error

**Reduce batch size**:
```bash
--batch_size 64  # or 32
```

**Reduce evaluation batch size**:
```bash
--eval_batch_size 32  # or 16
```

### Slow Validation

**Use fewer test pairs**:
```bash
--max_test_pairs 5000  # Faster validation
```

**Increase evaluation batch size** (if memory allows):
```bash
--eval_batch_size 128
```

### GPU Not Being Used

**Check CUDA availability**:
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Check GPU visibility**:
```bash
nvidia-smi
```

### Training Not Starting

**Check data paths in config**:
```yaml
train_list: /mnt/ricproject3/2025/data/train_list.txt
test_list: /mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt
train_path: /mnt/ricproject3/2025/data/rearranged_voxceleb2
test_path: /mnt/ricproject3/2025/data/rearranged_voxceleb1
```

**Verify files exist**:
```bash
ls -lh /mnt/ricproject3/2025/data/train_list.txt
ls -lh /mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt
```

---

## Performance Tips

### Speed Up Training

1. **Increase batch size** (if memory allows):
   ```bash
   --batch_size 256
   ```

2. **Use more DataLoader workers**:
   ```bash
   --nDataLoaderThread 16
   ```

3. **Reduce validation frequency**:
   ```bash
   --test_interval 5  # Instead of 3
   ```

4. **Use fewer validation pairs**:
   ```bash
   --max_test_pairs 5000
   ```

### Improve Accuracy

1. **Use more validation pairs**:
   ```bash
   --max_test_pairs 100000  # or 0 for all
   ```

2. **Reduce learning rate**:
   ```bash
   --lr 0.0005
   ```

3. **Enable data augmentation** (in config):
   ```yaml
   augment: true
   musan_path: /path/to/musan
   rir_path: /path/to/rirs
   ```

---

## Stopping Training

### Stop Gracefully (in tmux)

1. Attach to session: `tmux attach -t train_voxceleb`
2. Press `Ctrl+C` to stop training
3. Detach: `Ctrl+B` then `D`

### Kill tmux Session

```bash
tmux kill-session -t train_voxceleb
```

### Kill Process Directly

```bash
# Find process ID
ps aux | grep trainSpeakerNet

# Kill process (replace PID with actual number)
kill -9 <PID>
```

---

## Expected Training Time

Based on Tesla T4 GPU:

| Configuration | Time per Epoch | Validation Time | Total (100 epochs) |
|---------------|---------------|-----------------|-------------------|
| Default (10K pairs) | ~15 min | ~75 sec | ~27 hours |
| Fast (1K pairs) | ~15 min | ~10 sec | ~25 hours |
| Accurate (50K pairs) | ~15 min | ~6 min | ~35 hours |
| Full (553K pairs) | ~15 min | ~70 min | ~130 hours |

**Training samples**: 1,091,445  
**Speakers**: 5,991  
**Batch size**: 128  
**Epochs**: 500 (with early stopping)

---

## Performance Metrics

### During Training

Monitor these metrics in the log:

```
2025-10-25 18:41:16 Epoch 3, VEER 23.2935, MinDCF 0.83492, Threshold 0.000000
```

- **VEER**: Validation Equal Error Rate (lower is better)
- **MinDCF**: Minimum Detection Cost Function (lower is better)
- **Threshold**: Decision threshold for classification

### Best Model

Training automatically saves:
- Best model based on lowest EER
- Early stopping if no improvement for 15 test intervals (patience)

---

## Quick Reference

```bash
# Start training in background
tmux new-session -d -s train "cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer && ./run_performance_optimized.sh train"

# Check status
tmux attach -t train

# View log
tail -f training_*.log

# Check GPU
nvidia-smi

# Stop training
tmux kill-session -t train
```

---

## Additional Resources

- **MAX_TEST_PAIRS_GUIDE.md** - Guide for validation speed optimization
- **PERFORMANCE_README.md** - Detailed performance optimization documentation
- **PERFORMANCE_CHANGES_SUMMARY.md** - Summary of all optimizations
- **README_SL_COLVAI.md** - Project overview and setup

---

## Contact & Support

For issues specific to this optimized version, please check:
- GitHub: https://github.com/dimuthuanuraj/SL_ColvaiAI
- Log files in the training directory
- TensorBoard visualization

For original VoxCeleb trainer issues:
- Original repo: https://github.com/clovaai/voxceleb_trainer
