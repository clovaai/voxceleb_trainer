# Mini-VoxCeleb2 Dataset

A subset of VoxCeleb2 containing 140 randomly selected speakers for fast experimentation and development.

## Dataset Statistics

- **Speakers**: 140 (out of 5,991 total)
- **Audio Files**: 30,179
- **Total Size**: ~7.1 GB
- **Avg Files/Speaker**: 215.6
- **Selection Method**: Random sampling with seed=42 (reproducible)
- **Storage Method**: Symbolic links (no disk duplication)

## Created Files

| File | Description |
|------|-------------|
| `/mnt/ricproject3/2025/data/mini_voxceleb2/` | Mini dataset directory with 140 speaker folders |
| `/mnt/ricproject3/2025/data/mini_train_list.txt` | Training list with 30,179 entries |
| `/mnt/ricproject3/2025/data/mini_voxceleb2_speakers.txt` | List of 140 selected speaker IDs |
| `configs/mini_voxceleb2_config.yaml` | Pre-configured training config |

## Selected Speakers

The 140 speakers were randomly selected using seed=42 from the full VoxCeleb2 dataset.

View the complete list:
```bash
cat /mnt/ricproject3/2025/data/mini_voxceleb2_speakers.txt
```

Sample speaker IDs:
- id00084, id00332, id00353, id00389, id00412
- ... (135 more)
- id08901, id09048, id09067, id09230, id09235

## Usage

### Quick Training Test

```bash
cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer

python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb2_config.yaml \
  --save_path exps/mini_test_$(date +%Y%m%d_%H%M%S)
```

### Expected Training Time

With 2x Tesla T4 GPUs:
- **Epoch time**: ~3-4 minutes (vs ~15 min for full dataset)
- **Validation time**: ~75 seconds (10K pairs)
- **Total for 100 epochs**: ~6-7 hours (vs ~27+ hours for full)

**Training is approximately 4x faster** than the full VoxCeleb2 dataset.

### Configuration Differences

Compared to full VoxCeleb2 config:

| Parameter | Full Dataset | Mini Dataset | Reason |
|-----------|-------------|--------------|--------|
| `nClasses` | 5,991 | 140 | Number of speakers |
| `batch_size` | 128 | 64 | Smaller dataset |
| `nDataLoaderThread` | 8 | 6 | Less I/O needed |
| `max_epoch` | 500 | 100 | Faster overfitting |
| `train_list` | train_list.txt | mini_train_list.txt | Dataset path |
| `train_path` | rearranged_voxceleb2 | mini_voxceleb2 | Dataset path |

## When to Use Mini Dataset

✅ **Good for:**
- Quick experiments and hyperparameter tuning
- Testing new architectures or loss functions
- Development and debugging
- Fast iteration cycles
- Learning the training pipeline
- CI/CD testing

❌ **Not suitable for:**
- Final production models
- Publication-quality results
- Benchmarking against other methods
- Maximum accuracy requirements

## Recreating the Dataset

If you need to recreate with different speakers or quantity:

```bash
cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer

# Edit create_mini_voxceleb2.py to change:
# - NUM_SPEAKERS: number of speakers to select
# - SEED: random seed for different selection
# - SOURCE_DIR: source dataset path
# - TARGET_DIR: output directory

python3 create_mini_voxceleb2.py
```

### Customization Options

Edit `create_mini_voxceleb2.py`:

```python
# Configuration section
NUM_SPEAKERS = 140  # Change to 50, 100, 200, etc.
SEED = 42          # Change for different random selection
```

Then run:
```bash
python3 create_mini_voxceleb2.py
```

## Disk Space

The mini dataset uses **symbolic links**, so it consumes:
- **Actual disk usage**: ~4 KB (just the links)
- **Apparent size**: ~7.1 GB (pointed files)

To use actual copies instead of symlinks, edit `create_mini_voxceleb2.py`:
```python
# In create_mini_dataset function
stats = create_mini_dataset(SOURCE_DIR, TARGET_DIR, selected_speakers, 
                           use_symlinks=False)  # Change to False
```

⚠️ **Warning**: This will use 7.1 GB of actual disk space.

## Validation

The mini dataset still uses the **full VoxCeleb1 test set** for validation:
- Test set: 553,550 pairs (unchanged)
- Test path: `/mnt/ricproject3/2025/data/rearranged_voxceleb1`
- Test list: `/mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt`

This ensures validation results are comparable to full-dataset training.

## Performance Comparison

Expected performance differences:

| Metric | Full Dataset | Mini Dataset | Notes |
|--------|-------------|--------------|-------|
| Training Speed | 1x | 4x faster | Less data to process |
| EER | ~15-20% | ~25-35% | Less speaker diversity |
| Convergence | ~100 epochs | ~30-50 epochs | Faster overfitting |
| Model Size | Same | Same | Architecture unchanged |

## Directory Structure

```
/mnt/ricproject3/2025/data/
├── rearranged_voxceleb2/          # Original full dataset (5,991 speakers)
├── mini_voxceleb2/                # Mini dataset (140 speakers, symlinks)
│   ├── id00084/ -> ../rearranged_voxceleb2/id00084/
│   ├── id00332/ -> ../rearranged_voxceleb2/id00332/
│   └── ... (140 total)
├── train_list.txt                 # Original training list (1,091,445 files)
├── mini_train_list.txt            # Mini training list (30,179 files)
└── mini_voxceleb2_speakers.txt    # List of selected speakers
```

## Verifying the Dataset

Check speaker count:
```bash
ls -d /mnt/ricproject3/2025/data/mini_voxceleb2/id* | wc -l
# Should output: 140
```

Check training list:
```bash
wc -l /mnt/ricproject3/2025/data/mini_train_list.txt
# Should output: 30179
```

Verify symbolic links:
```bash
ls -lh /mnt/ricproject3/2025/data/mini_voxceleb2/ | head
# Should show symlinks to rearranged_voxceleb2
```

## Training Examples

### Quick Test (5 epochs)
```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb2_config.yaml \
  --max_epoch 5 \
  --save_path exps/quick_test
```

### Development Training
```bash
tmux new-session -d -s "mini_train" \
"python3 trainSpeakerNet_performance_updated.py \
--config configs/mini_voxceleb2_config.yaml \
--save_path exps/mini_dev_$(date +%Y%m%d_%H%M%S) \
2>&1 | tee training_mini_$(date +%Y%m%d_%H%M%S).log; exec bash"
```

### With Different Learning Rate
```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb2_config.yaml \
  --lr 0.0005 \
  --save_path exps/mini_lr0005
```

### Transfer Learning Test
```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb2_config.yaml \
  --initial_model /mnt/ricproject3/2025/data/baseline_v2_smproto.model \
  --save_path exps/mini_transfer
```

## Troubleshooting

### Error: nClasses mismatch

If you see an error about class numbers:
```
RuntimeError: nClasses in config (140) doesn't match model (5991)
```

**Solution**: Make sure `nClasses: 140` in your config file.

### Error: FileNotFoundError

If training can't find files:
```
FileNotFoundError: /mnt/ricproject3/2025/data/rearranged_voxceleb2/id00xxx/...
```

**Check**:
1. Symbolic links are valid: `ls -lh /mnt/ricproject3/2025/data/mini_voxceleb2/id00084`
2. Source dataset exists: `ls /mnt/ricproject3/2025/data/rearranged_voxceleb2/`

### Slow Training

If training is slower than expected:
- Check batch size: Try increasing `--batch_size 128`
- Check GPU usage: `nvidia-smi`
- Reduce augmentation: Set `augment: false` in config
- Use fewer workers: Set `nDataLoaderThread: 4`

## Cleanup

To remove the mini dataset:

```bash
# Remove dataset directory (symlinks only, original data safe)
rm -rf /mnt/ricproject3/2025/data/mini_voxceleb2

# Remove training list
rm /mnt/ricproject3/2025/data/mini_train_list.txt

# Remove speaker list
rm /mnt/ricproject3/2025/data/mini_voxceleb2_speakers.txt
```

**Note**: This only removes symlinks, not the original VoxCeleb2 data.

## License

Same as original VoxCeleb2 dataset. For research and educational use only.

## References

- Full VoxCeleb2 Dataset: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html
- Original Paper: Chung et al., "VoxCeleb2: Deep Speaker Recognition", INTERSPEECH 2018
- This subset created for fast experimentation purposes

## Support

For issues with mini dataset creation or configuration:
1. Check this README
2. Verify file paths in config
3. Check log files for errors
4. See TRAINING_GUIDE.md for general training help

---

**Created**: October 28, 2025  
**Tool**: `create_mini_voxceleb2.py`  
**Seed**: 42 (reproducible)
