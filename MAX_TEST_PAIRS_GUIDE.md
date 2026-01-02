# max_test_pairs Parameter Guide

## Overview
The `max_test_pairs` parameter allows you to control how many test pairs are used during validation, making validation faster during training while still getting meaningful feedback.

## Usage

### In Config File
Edit `configs/experiment_01_performance_updated.yaml`:
```yaml
max_test_pairs: 10000  # Use 10,000 pairs
```

### Command Line Override
```bash
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --max_test_pairs 5000
```

### Special Values
- `0` = Use ALL test pairs (full validation)
- `> 0` = Use first N pairs from test list

## Recommended Values

| max_test_pairs | Validation Time | Use Case |
|----------------|-----------------|----------|
| 1,000 | ~10 seconds | Quick test / debugging |
| 5,000 | ~40 seconds | Fast feedback during training |
| 10,000 | ~75 seconds | **Default** - good balance |
| 50,000 | ~6 minutes | More accurate metrics |
| 100,000 | ~13 minutes | High accuracy |
| 0 (all 553,550) | ~70 minutes | Final evaluation / paper results |

## Speed Estimates

Based on Tesla T4 GPU performance (~130 pairs/second):

- **1K pairs**: 7-10 seconds
- **10K pairs**: 70-80 seconds  
- **100K pairs**: 12-15 minutes
- **Full (553K)**: 65-75 minutes

## Examples

### Quick Training with Fast Validation
```bash
# Validate every 3 epochs with 5,000 pairs (~40 seconds)
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --max_test_pairs 5000
```

### Full Evaluation
```bash
# Use all 553,550 pairs for final evaluation
python3 trainSpeakerNet_performance_updated.py \
  --config configs/experiment_01_performance_updated.yaml \
  --max_test_pairs 0
```

### Progressive Validation
```yaml
# Start fast, gradually increase
# Epochs 1-10: 5,000 pairs
# Epochs 11-30: 20,000 pairs  
# Epochs 31+: Full validation
```

## Testing the Parameter

Use the test script to verify settings:
```bash
python3 test_max_pairs_param.py
python3 test_max_pairs_param.py --max_test_pairs 5000
```

## Notes

1. **Consistent Pairs**: The same first N pairs are always used, ensuring consistent comparison across epochs
2. **EER Impact**: Metrics from subset may differ from full validation by 0.1-0.5% EER
3. **Training Speed**: Reducing validation time significantly speeds up overall training
4. **Memory**: Using fewer pairs doesn't reduce memory usage significantly (embeddings are cached)

## Validation Time Breakdown

For 10,000 pairs:
- Load unique audio files: ~15 seconds
- Compute embeddings: ~40 seconds
- Compute scores: ~15 seconds  
- Calculate metrics: ~5 seconds
- **Total**: ~75 seconds

For full 553,550 pairs:
- Load unique audio files: ~15 seconds
- Compute embeddings: ~40 seconds
- Compute scores: ~65 minutes
- Calculate metrics: ~5 seconds
- **Total**: ~70 minutes

## Best Practices

### Development Phase
```yaml
max_test_pairs: 5000  # Fast iteration
test_interval: 3      # Frequent validation
```

### Production Training
```yaml
max_test_pairs: 50000  # Good accuracy
test_interval: 5       # Less frequent
```

### Final Evaluation
```yaml
max_test_pairs: 0      # Full validation
test_interval: 10      # Infrequent, thorough
```

## See Also

- `quick_test_validation.py` - Quick validation test script
- `test_max_pairs_param.py` - Parameter verification script
- `experiment_01_performance_updated.yaml` - Config file with parameter
