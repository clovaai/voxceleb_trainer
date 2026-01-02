# Mini-VoxCeleb1 Dataset

A subset of VoxCeleb1 containing 50 randomly selected speakers for fast experimentation and development.

## Dataset Statistics

- **Speakers**: 50 (out of 1,211 total)
- **Audio Files**: 6,286
- **Total Size**: ~1.6 GB
- **Avg Files/Speaker**: 125.7
- **Selection Method**: Random sampling with seed=42 (reproducible)
- **Storage Method**: Symbolic links (no disk duplication)

## Test Pairs Statistics

⚠️ **Important Note**: The mini test set has an **imbalanced distribution**:
- **Total pairs**: 12,559
- **Positive pairs (same speaker)**: 12,094 (96.3%)
- **Negative pairs (different speaker)**: 465 (3.7%)

This imbalance occurs because we only kept pairs where **both speakers** are in the selected 50 speakers. With fewer speakers, there are far fewer combinations for negative pairs.

**Recommendation**: For accurate testing, use the **full VoxCeleb1 test set** even when training on mini dataset.

## Created Files

| File | Description |
|------|-------------|
| `/mnt/ricproject3/2025/data/mini_voxceleb1/` | Mini dataset directory with 50 speaker folders |
| `/mnt/ricproject3/2025/data/mini_voxceleb1_train_list.txt` | Training list with 6,286 entries |
| `/mnt/ricproject3/2025/data/mini_test_list.txt` | Test list with 12,559 pairs (⚠️ imbalanced) |
| `/mnt/ricproject3/2025/data/mini_voxceleb1_speakers.txt` | List of 50 selected speaker IDs |
| `configs/mini_voxceleb1_config.yaml` | Pre-configured training config |

## Selected Speakers

The 50 speakers were randomly selected using seed=42 from the full VoxCeleb1 dataset.

View the complete list:
```bash
cat /mnt/ricproject3/2025/data/mini_voxceleb1_speakers.txt
```

Sample speaker IDs:
- id10014, id10052, id10055, id10062, id10066
- ... (45 more)
- id11171, id11190, id11223, id11247, id11250

## Usage

### Quick Training Test (Mini Train, Mini Test)

```bash
cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer

python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb1_config.yaml \
  --save_path exps/mini_vox1_test_$(date +%Y%m%d_%H%M%S)
```

⚠️ **Warning**: Using mini test set will give **optimistic results** due to imbalance.

### Recommended: Train Mini, Test Full (More Realistic)

```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb1_config.yaml \
  --test_list /mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt \
  --test_path /mnt/ricproject3/2025/data/rearranged_voxceleb1 \
  --max_test_pairs 10000 \
  --save_path exps/mini_train_full_test_$(date +%Y%m%d_%H%M%S)
```

This uses:
- **Training**: Mini-VoxCeleb1 (50 speakers, fast)
- **Testing**: Full VoxCeleb1 test set (accurate evaluation)

## Expected Training Time

With 2x Tesla T4 GPUs:

| Configuration | Epoch Time | Validation Time | Total (100 epochs) |
|---------------|------------|-----------------|-------------------|
| Mini train + Mini test | ~1-2 min | ~2 min | ~5-6 hours |
| Mini train + Full test (10K pairs) | ~1-2 min | ~75 sec | ~4-5 hours |
| Mini train + Full test (all pairs) | ~1-2 min | ~70 min | ~120 hours |

**Training is approximately 7-10x faster** than the full VoxCeleb2 dataset.

## Configuration Differences

Compared to full dataset configs:

| Parameter | Full VoxCeleb1 | Mini VoxCeleb1 | Reason |
|-----------|---------------|----------------|--------|
| `nClasses` | 1,211 | 50 | Number of speakers |
| `batch_size` | 128 | 32 | Much smaller dataset |
| `nDataLoaderThread` | 8 | 4 | Less I/O needed |
| `max_epoch` | 500 | 100 | Faster overfitting |
| `train_list` | N/A | mini_voxceleb1_train_list.txt | Dataset path |
| `test_list` | list_test_all_formated_cleaned.txt | mini_test_list.txt | Test pairs |
| `train_path` | rearranged_voxceleb1 | mini_voxceleb1 | Dataset path |
| `test_path` | rearranged_voxceleb1 | mini_voxceleb1 | Dataset path |

## When to Use Mini VoxCeleb1

✅ **Good for:**
- Ultra-fast experiments (smallest dataset)
- Testing code changes quickly
- Development and debugging
- Learning the training pipeline
- CI/CD testing
- Quick hyperparameter exploration

❌ **Not suitable for:**
- Accurate model evaluation (use full test set)
- Final production models
- Publication-quality results
- Benchmarking performance

## Comparison: Mini VoxCeleb1 vs Mini VoxCeleb2

| Metric | Mini VoxCeleb1 | Mini VoxCeleb2 |
|--------|----------------|----------------|
| Speakers | 50 | 140 |
| Audio Files | 6,286 | 30,179 |
| Size | 1.6 GB | 7.1 GB |
| Avg Files/Speaker | 125.7 | 215.6 |
| Epoch Time | 1-2 min | 3-4 min |
| Use Case | Quick testing | More realistic training |

**Use Mini VoxCeleb1 when**: You need the absolute fastest iteration cycle  
**Use Mini VoxCeleb2 when**: You need more training data for better convergence

## Combined Mini Training (Both Datasets)

You can combine both mini datasets for more speakers:

```yaml
# In config file
nClasses: 190  # 50 + 140 speakers

# Combine train lists
train_list: /path/to/combined_mini_train_list.txt
```

Create combined list:
```bash
cat /mnt/ricproject3/2025/data/mini_voxceleb1_train_list.txt \
    /mnt/ricproject3/2025/data/mini_train_list.txt \
    > /mnt/ricproject3/2025/data/combined_mini_train_list.txt
```

## Recreating the Dataset

If you need to recreate with different speakers or quantity:

```bash
cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer

# Edit create_mini_voxceleb1.py to change:
# - NUM_SPEAKERS: number of speakers to select
# - SEED: random seed for different selection

python3 create_mini_voxceleb1.py
```

### Customization Options

Edit `create_mini_voxceleb1.py`:

```python
# Configuration section
NUM_SPEAKERS = 50  # Change to 20, 100, etc.
SEED = 42         # Change for different random selection
```

## Disk Space

The mini dataset uses **symbolic links**, so it consumes:
- **Actual disk usage**: ~4 KB (just the links)
- **Apparent size**: ~1.6 GB (pointed files)

## Validation Considerations

### Why the Test Set is Imbalanced

The mini test set filters the full VoxCeleb1 test pairs to only include pairs where **both files** belong to the selected 50 speakers.

- Original test set: 553,550 pairs (50% positive, 50% negative)
- Mini test set: 12,559 pairs (96.3% positive, 3.7% negative)

**Why this happens**:
- Positive pairs: Only need 1 speaker to match → Many pairs retained
- Negative pairs: Need 2 different speakers, both in mini set → Very few combinations

### Recommended Testing Strategies

**Option 1: Use full test set** (Recommended)
```bash
--test_list /mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt
--test_path /mnt/ricproject3/2025/data/rearranged_voxceleb1
--max_test_pairs 10000  # Use subset for speed
```

**Option 2: Accept the limitation**
- Use mini test set for quick sanity checks only
- Understand results will be optimistic
- Always validate with full test set before conclusions

**Option 3: Create balanced mini test set**
- Manually select equal positive/negative pairs
- Requires custom script (not provided)

## Training Examples

### Ultra-Fast Test (5 epochs, mini test)
```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb1_config.yaml \
  --max_epoch 5 \
  --save_path exps/ultra_fast_test
```

### Development Training (mini test)
```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb1_config.yaml \
  --save_path exps/mini_vox1_dev_$(date +%Y%m%d_%H%M%S)
```

### Realistic Training (full test set)
```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb1_config.yaml \
  --test_list /mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt \
  --test_path /mnt/ricproject3/2025/data/rearranged_voxceleb1 \
  --max_test_pairs 10000 \
  --save_path exps/mini_train_full_test
```

### Transfer Learning Test
```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb1_config.yaml \
  --initial_model /mnt/ricproject3/2025/data/baseline_v2_smproto.model \
  --save_path exps/mini_vox1_transfer
```

## Directory Structure

```
/mnt/ricproject3/2025/data/
├── rearranged_voxceleb1/              # Original full dataset (1,211 speakers)
├── mini_voxceleb1/                    # Mini dataset (50 speakers, symlinks)
│   ├── id10014/ -> ../rearranged_voxceleb1/id10014/
│   ├── id10052/ -> ../rearranged_voxceleb1/id10052/
│   └── ... (50 total)
├── list_test_all_formated_cleaned.txt # Full test pairs (553,550)
├── mini_test_list.txt                 # Mini test pairs (12,559, imbalanced)
├── mini_voxceleb1_train_list.txt      # Mini training list (6,286 files)
└── mini_voxceleb1_speakers.txt        # List of selected speakers
```

## Verifying the Dataset

Check speaker count:
```bash
ls -d /mnt/ricproject3/2025/data/mini_voxceleb1/id* | wc -l
# Should output: 50
```

Check training list:
```bash
wc -l /mnt/ricproject3/2025/data/mini_voxceleb1_train_list.txt
# Should output: 6286
```

Check test list:
```bash
wc -l /mnt/ricproject3/2025/data/mini_test_list.txt
# Should output: 12559
```

Check test set balance:
```bash
grep "^1 " /mnt/ricproject3/2025/data/mini_test_list.txt | wc -l  # Positive pairs
grep "^0 " /mnt/ricproject3/2025/data/mini_test_list.txt | wc -l  # Negative pairs
```

Verify symbolic links:
```bash
ls -lh /mnt/ricproject3/2025/data/mini_voxceleb1/ | head
# Should show symlinks to rearranged_voxceleb1
```

## Troubleshooting

### Error: nClasses mismatch

If you see an error about class numbers:
```
RuntimeError: nClasses in config (50) doesn't match model (1211)
```

**Solution**: Make sure `nClasses: 50` in your config file.

### Poor EER Results

If EER seems too good (~5-10%):
- You're likely using the **imbalanced mini test set**
- Switch to full test set for realistic evaluation
- Mini test has 96.3% positive pairs (easier to get good scores)

### Training Too Fast

If training finishes in minutes:
- This is expected! Only 6,286 files
- Each epoch is ~1-2 minutes
- Consider using mini-VoxCeleb2 for more training data

## Performance Expectations

### With Mini Test Set (Imbalanced)
- **EER**: 10-20% (optimistic due to imbalance)
- **Warning**: Results not comparable to standard benchmarks

### With Full Test Set (Realistic)
- **EER**: 35-45% (realistic for 50 speakers)
- Comparable to standard evaluation
- Much worse than full training (expected)

## Cleanup

To remove the mini dataset:

```bash
# Remove dataset directory (symlinks only, original data safe)
rm -rf /mnt/ricproject3/2025/data/mini_voxceleb1

# Remove training list
rm /mnt/ricproject3/2025/data/mini_voxceleb1_train_list.txt

# Remove test list
rm /mnt/ricproject3/2025/data/mini_test_list.txt

# Remove speaker list
rm /mnt/ricproject3/2025/data/mini_voxceleb1_speakers.txt
```

**Note**: This only removes symlinks, not the original VoxCeleb1 data.

## License

Same as original VoxCeleb1 dataset. For research and educational use only.

## References

- Full VoxCeleb1 Dataset: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
- Original Paper: Nagrani et al., "VoxCeleb: A Large-Scale Speaker Identification Dataset", INTERSPEECH 2017
- This subset created for fast experimentation purposes

## Support

For issues with mini dataset creation or configuration:
1. Check this README
2. Verify file paths in config
3. Check log files for errors
4. See TRAINING_GUIDE.md for general training help
5. See MINI_VOXCELEB2_README.md for comparison

---

**Created**: October 28, 2025  
**Tool**: `create_mini_voxceleb1.py`  
**Seed**: 42 (reproducible)  
**Warning**: Mini test set is highly imbalanced - use full test set for accurate evaluation
