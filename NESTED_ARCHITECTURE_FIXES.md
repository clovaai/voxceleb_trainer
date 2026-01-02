# Nested Architecture Stability Fixes

**Date**: December 29, 2025  
**Problem**: Training collapsed at Epoch 11 with NaN loss and degraded to 100% EER

## Root Cause Analysis

1. **Gradient Explosion**: Multiple nested paths (2^4 = 16 gradient paths) caused accumulation
2. **Numerical Instability**: Bilinear interpolation introduced errors when aligning features
3. **Fixed Scaling**: 0.5× multiplier was insufficient to prevent instability
4. **BatchNorm Sensitivity**: BatchNorm with small batches (32) unstable for nested connections

## Fixes Applied

### 1. Learnable Nested Weights ✅
**Problem**: Fixed 0.5× scaling couldn't adapt to different levels
**Solution**:
```python
self.nested_weight = nn.Parameter(torch.tensor(0.3))
weight = torch.sigmoid(self.nested_weight)  # Constrain to [0, 1]
x = x + weight * prev_info
```
**Benefit**: Each level learns optimal aggregation weight dynamically

### 2. Replace Interpolation with Adaptive Pooling ✅
**Problem**: Bilinear interpolation unstable for gradient flow
**Solution**:
```python
# Before: F.interpolate(feat, size=x.shape[2:], mode='bilinear')
# After:  F.adaptive_avg_pool2d(feat, (target_h, target_w))
```
**Benefit**: More stable downsampling, better gradient properties

### 3. GroupNorm Instead of BatchNorm ✅
**Problem**: BatchNorm unstable with batch_size=32
**Solution**:
```python
# Before: nn.BatchNorm2d(channels)
# After:  nn.GroupNorm(8, channels)  # 8 groups
```
**Benefit**: Normalization independent of batch size, more stable gradients

### 4. Add Dropout for Regularization ✅
**Solution**:
```python
nn.Dropout2d(0.1)  # After nested aggregation
```
**Benefit**: Prevents overfitting to nested connections, better generalization

### 5. Increase Batch Size ✅
**Before**: 32  
**After**: 48 (+50%)  
**Benefit**: More stable gradient estimates, better GroupNorm statistics

### 6. Improved Hyperparameters ✅
| Parameter | Before | After | Reason |
|-----------|--------|-------|---------|
| **LR** | 0.0005 | 0.0008 | Faster convergence with stability |
| **LR Decay** | 0.95 | 0.97 | More gradual, prevents collapse |
| **Weight Decay** | 1e-4 | 2e-4 | Stronger regularization |
| **Patience** | 15 | 20 | Allow more recovery time |
| **Max Epochs** | 100 | 150 | Full convergence opportunity |

### 7. Gradient Clipping (Already Present) ✅
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```
**Benefit**: Prevents extreme gradients from any single batch

## Architecture Changes Summary

```python
# Key improvements in NestedBlock:
1. Learnable weights: self.nested_weight (initialized 0.3, constrained [0,1])
2. Adaptive pooling: F.adaptive_avg_pool2d() instead of F.interpolate()
3. GroupNorm: nn.GroupNorm(8, channels) instead of BatchNorm
4. Dropout: nn.Dropout2d(0.1) for regularization
5. Stable aggregation: sigmoid(weight) * prev_info
```

## Expected Improvements

| Metric | Before Fixes | After Fixes (Expected) |
|--------|--------------|----------------------|
| **Training Stability** | Collapsed at epoch 11 | Stable to 150 epochs |
| **NaN Occurrence** | Yes (epoch 11) | None |
| **Best EER** | 21.72% (unstable) | ~14-16% (stable) |
| **Convergence** | Failed | Full convergence |
| **Gradient Flow** | Unstable | Smooth and bounded |

## Validation Checklist

- [x] Architecture compiles without errors
- [x] Forward pass works (test_nested_architecture.py passed)
- [x] Inference speed maintained (28.83ms, 1.09× faster than baseline)
- [x] Training started successfully
- [ ] No NaN after 15 epochs
- [ ] EER improves beyond 21.72%
- [ ] Model converges to <16% EER
- [ ] Stable training to 100+ epochs

## Monitoring During Training

Watch for:
1. **Loss trajectory**: Should decrease smoothly without sudden spikes
2. **Nested weights**: Check TensorBoard for learned weight values (should stay 0.2-0.6)
3. **Gradient norms**: Should stay below 5.0 (clipping threshold)
4. **EER progression**: Should improve beyond 21.72% by epoch 20
5. **No NaN**: Loss should never become NaN

## Rollback Plan (If Still Fails)

If training still collapses:
1. **Reduce nested connections**: Only Level 3 receives all previous (not all levels)
2. **Simpler architecture**: Remove multi-scale fusion, use only final level
3. **Switch to proven baseline**: Use ResNetSE34L with ASP encoder (15.48% EER proven)
4. **Try LSTM+Autoencoder**: The other paper approach (more stable for audio)

## Files Modified

1. `models/NestedSpeakerNet.py`: Architecture stability fixes
2. `configs/nested_4level.yaml`: Hyperparameter tuning
3. Training command: Batch size 48, new log file

## Next Steps

1. **Monitor training** for 30 minutes (first 15-20 epochs critical)
2. **Check scores.txt** at epoch 15 - should show EER < 21.72%
3. **Validate stability** - no NaN by epoch 30
4. **Compare with baseline** - target <15.48% EER to beat ResNetSE34L
5. **Document results** for research paper

## Commands to Monitor

```bash
# Watch training progress
tail -f logs/nested_fixed_*.log

# Check current EER
tail -20 exps/nested_4level_exp1/result/scores.txt

# Monitor GPU usage
nvidia-smi -l 1

# Check learned nested weights (after 10+ epochs)
# Will show in TensorBoard or model inspection
```

---
**Status**: Training in progress (PID 33686)  
**Log**: `logs/nested_fixed_20251229_*.log`  
**Experiment**: `exps/nested_4level_exp1/`
