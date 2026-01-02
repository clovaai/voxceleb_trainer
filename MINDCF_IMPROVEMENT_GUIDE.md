# MinDCF Improvement Guide for VoxCeleb Speaker Verification

**Date:** October 30, 2025  
**Current Setup:** ResNetSE34L + SAP + AAMSoftmax  
**Goal:** Improve MinDCF (Minimum Detection Cost Function) performance

---

## Current Configuration Analysis

**Your Current Setup:**
- **Model:** ResNetSE34L (Large ResNet with Squeeze-and-Excitation)
- **Encoder:** SAP (Self-Attentive Pooling)
- **Loss:** AAMSoftmax (Additive Angular Margin Softmax)
- **Embedding Size:** 512
- **Features:** 80 mel-filterbanks
- **Margin:** 0.2, Scale: 30

---

## 🎯 Strategy 1: Model Architecture Improvements

### 1.1 Try Different Models (Ranked by Expected Performance)

#### **Best Option: RawNet3 (State-of-the-Art)**
```yaml
model: RawNet3
encoder_type: ECA  # or ASP
nOut: 256  # or 512
log_sinc: true
norm_sinc: true
out_bn: false
sinc_stride: 10
```

**Why RawNet3?**
- ✅ Learns filterbanks directly from raw waveform (no mel-spectrogram)
- ✅ End-to-end learnable features
- ✅ State-of-the-art performance on VoxCeleb
- ✅ Better noise robustness
- ⚠️ Requires more GPU memory
- ⚠️ Slower training (but better results)

**Expected Improvement:** 10-20% relative MinDCF reduction

---

#### **Good Option: ResNetSE34V2**
```yaml
model: ResNetSE34V2
encoder_type: ASP  # Attentive Statistics Pooling
nOut: 512
n_mels: 80
```

**Why ResNetSE34V2?**
- ✅ Improved SE blocks compared to ResNetSE34L
- ✅ Better gradient flow
- ✅ Similar speed to your current model
- ✅ Good balance of performance and efficiency

**Expected Improvement:** 5-10% relative MinDCF reduction

---

#### **Experimental Option: VGGVox**
```yaml
model: VGGVox
encoder_type: SAP
nOut: 512
n_mels: 80
```

**Why VGGVox?**
- ✅ Lightweight and fast
- ✅ Good baseline model
- ⚠️ Lower performance than ResNet variants
- ⚠️ Better for deployment than accuracy

**Expected Improvement:** May be worse than ResNetSE34L

---

### 1.2 Encoder Type Optimization

Your current encoder is **SAP** (Self-Attentive Pooling). Try these alternatives:

#### **ASP (Attentive Statistics Pooling)** - RECOMMENDED
```yaml
encoder_type: ASP
```

**Advantages:**
- ✅ Captures both mean AND variance (2x feature dimension)
- ✅ Better temporal modeling
- ✅ Usually outperforms SAP by 5-10%
- ✅ No architectural changes needed

**Changes Required:**
- Output dimension doubles automatically
- May need to adjust `nOut` if memory constrained

**Expected Improvement:** 5-10% relative MinDCF reduction

---

## 🎯 Strategy 2: Loss Function Improvements

### 2.1 Advanced Loss Functions (Ranked)

#### **Option A: Margin Tuning (AAMSoftmax)** - EASIEST
Keep AAMSoftmax but optimize margins:

```yaml
trainfunc: aamsoftmax
margin: 0.3  # Increase from 0.2
scale: 32    # Increase from 30
```

**Margin Tuning Guidelines:**
```yaml
# Conservative (safer)
margin: 0.25
scale: 30

# Aggressive (better separation)
margin: 0.35
scale: 35

# Very aggressive (may hurt generalization)
margin: 0.4
scale: 40
```

**Expected Improvement:** 3-8% relative MinDCF reduction

---

#### **Option B: Sub-center AAMSoftmax** - BEST PERFORMANCE
```yaml
trainfunc: subcenter_aamsoftmax  # If implemented
margin: 0.2
scale: 30
K: 3  # Number of sub-centers per class
```

**Advantages:**
- ✅ Handles intra-class variance better
- ✅ More robust to noisy labels
- ✅ Better calibration
- ⚠️ Requires implementation if not available

**Expected Improvement:** 10-15% relative MinDCF reduction

---

#### **Option C: Triplet Loss** - FOR SMALL DATASETS
```yaml
trainfunc: triplet
margin: 0.2
```

**Advantages:**
- ✅ Good for small datasets (like your mini_voxceleb1)
- ✅ Direct metric learning
- ⚠️ Requires careful mining strategy
- ⚠️ Slower convergence

**Expected Improvement:** 5-10% on small datasets

---

#### **Option D: Prototypical Loss**
```yaml
trainfunc: proto  # or angleproto
```

**Advantages:**
- ✅ Few-shot learning friendly
- ✅ Good for unbalanced datasets
- ⚠️ May underperform on large datasets

**Expected Improvement:** Depends on dataset balance

---

## 🎯 Strategy 3: Feature Engineering

### 3.1 Mel-Filterbank Optimization

#### **Increase Frequency Resolution**
```yaml
n_mels: 120  # Increase from 80
```

**Trade-offs:**
- ✅ More frequency information
- ✅ Better speaker discrimination
- ⚠️ More computation
- ⚠️ More GPU memory

**Expected Improvement:** 2-5% relative MinDCF reduction

---

#### **Enable Logarithmic Input**
```yaml
log_input: true  # Currently false
```

**Advantages:**
- ✅ Better dynamic range handling
- ✅ More robust to volume variations
- ✅ Standard practice in speaker recognition

**Expected Improvement:** 3-7% relative MinDCF reduction

---

### 3.2 Input Augmentation

Your current config has augmentation enabled but can be optimized:

```yaml
augment: true
musan_path: /mnt/ricproject3/2025/data/musan
rir_path: /mnt/ricproject3/2025/data/RIRS_NOISES/simulated_rirs

# Add these parameters:
augment_chain:
  - noise: 0.3      # 30% probability
  - reverb: 0.3     # 30% probability  
  - music: 0.2      # 20% probability
  - speech: 0.2     # 20% probability
```

**Expected Improvement:** 5-10% relative MinDCF reduction

---

## 🎯 Strategy 4: Training Strategy Improvements

### 4.1 Embedding Size Optimization

#### **Increase Embedding Dimension**
```yaml
nOut: 512  # Current (good)
# Try:
nOut: 768  # Better discrimination
# or
nOut: 1024  # Best discrimination (more memory)
```

**Guidelines:**
- Small datasets (< 1000 speakers): 256-512
- Medium datasets (1000-5000 speakers): 512-768
- Large datasets (> 5000 speakers): 768-1024

**Expected Improvement:** 3-8% with larger embeddings

---

### 4.2 Learning Rate Scheduling

#### **Cosine Annealing** - RECOMMENDED
```yaml
scheduler: cosine
lr: 0.001
min_lr: 0.00001
T_max: 100  # Total epochs
```

**Advantages:**
- ✅ Smooth convergence
- ✅ Better final performance
- ✅ No manual tuning needed

---

#### **Warm-up + Decay**
```yaml
scheduler: warmup_cosine
warmup_epochs: 5
lr: 0.001
min_lr: 0.00001
```

**Expected Improvement:** 2-5% relative MinDCF reduction

---

### 4.3 Optimizer Improvements

#### **Try AdamW** - RECOMMENDED
```yaml
optimizer: adamw
lr: 0.001
weight_decay: 0.0001  # L2 regularization
```

**Advantages:**
- ✅ Better generalization than Adam
- ✅ Decoupled weight decay
- ✅ Often better MinDCF

**Expected Improvement:** 2-5% relative MinDCF reduction

---

#### **Try SGD with Momentum** (for larger datasets)
```yaml
optimizer: sgd
lr: 0.01  # Higher LR for SGD
momentum: 0.9
weight_decay: 0.0001
```

**Expected Improvement:** 3-7% on large datasets

---

### 4.4 Batch Size & nPerSpeaker

#### **Optimize nPerSpeaker**
```yaml
nPerSpeaker: 2  # Current is 1
batch_size: 32  # Keep or reduce to 16 if memory issues
```

**Guidelines:**
```yaml
# For AAMSoftmax:
nPerSpeaker: 1  # Standard, fast
nPerSpeaker: 2  # Better, note TEER/TAcc bug (cosmetic)

# For Triplet/Contrastive:
nPerSpeaker: 2  # Required minimum
nPerSpeaker: 3  # Better triplet mining
```

**Expected Improvement:** 5-10% with nPerSpeaker=2

---

## 🎯 Strategy 5: Data-Centric Improvements

### 5.1 Training Data Optimization

#### **Use More Training Data**
If you're using mini_voxceleb2 (140 speakers), switch to full:

```yaml
train_list: /mnt/ricproject2/voxceleb_new/voxceleb2_train.txt
train_path: /mnt/ricproject2/voxceleb_new/voxceleb2/train/wav
nClasses: 5994  # Full VoxCeleb2
```

**Expected Improvement:** 20-30% relative MinDCF reduction (huge!)

---

### 5.2 Evaluation Optimization

#### **Use Full Test Pairs**
```yaml
max_test_pairs: 0  # Already optimal (use all pairs)
```

#### **Increase Evaluation Frames**
```yaml
eval_frames: 300  # Instead of 0 (variable length)
```

**Trade-off:**
- Variable length (0): Most accurate but slower
- Fixed length (300): Faster but may hurt accuracy slightly

---

### 5.3 Audio Quality

#### **Check Audio Preprocessing**
Ensure consistent preprocessing:
- Sample rate: 16 kHz
- No silence removal (or consistent across train/test)
- No aggressive VAD

---

## 🎯 Strategy 6: Score Normalization

### Post-Processing Improvements

#### **Implement Score Normalization** (not in current code)
```python
# Adaptive S-Norm (Symmetric Normalization)
def adaptive_snorm(scores, cohort_scores, top_n=200):
    """
    Normalize scores using cohort statistics
    """
    # Select top-N cohort scores
    cohort_sorted = np.sort(cohort_scores)[::-1][:top_n]
    
    # Compute statistics
    mean = np.mean(cohort_sorted)
    std = np.std(cohort_sorted)
    
    # Normalize
    normalized = (scores - mean) / (std + 1e-6)
    return normalized
```

**Expected Improvement:** 10-20% relative MinDCF reduction

---

## 📊 Recommended Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
**Minimal code changes, good improvements:**

1. ✅ **Enable log_input**
   ```yaml
   log_input: true
   ```
   Expected: 3-7% improvement

2. ✅ **Switch to ASP encoder**
   ```yaml
   encoder_type: ASP
   ```
   Expected: 5-10% improvement

3. ✅ **Tune margins**
   ```yaml
   margin: 0.3
   scale: 32
   ```
   Expected: 5-8% improvement

4. ✅ **Set nPerSpeaker=2**
   ```yaml
   nPerSpeaker: 2
   ```
   Expected: 5-10% improvement

**Total Expected: 15-30% relative MinDCF reduction** 🎯

---

### Phase 2: Model Upgrade (3-5 days)
**Requires testing different architectures:**

5. ✅ **Try ResNetSE34V2**
   ```yaml
   model: ResNetSE34V2
   encoder_type: ASP
   ```
   Expected: 5-10% additional improvement

6. ✅ **Try RawNet3** (if you have resources)
   ```yaml
   model: RawNet3
   encoder_type: ECA
   ```
   Expected: 10-20% additional improvement

---

### Phase 3: Advanced (1 week)
**Requires implementation:**

7. ✅ **Implement Sub-center AAMSoftmax**
   Expected: 10-15% additional improvement

8. ✅ **Add Score Normalization**
   Expected: 10-20% additional improvement

9. ✅ **Train on Full VoxCeleb2** (if possible)
   Expected: 20-30% additional improvement

---

## 🧪 Experimental Config: Best Setup

Here's my recommended config for best MinDCF:

```yaml
## Model - BEST PERFORMANCE
model: RawNet3  # or ResNetSE34V2 for faster training
encoder_type: ECA  # or ASP
nOut: 512
n_mels: 80  # Not used for RawNet3
log_input: true
log_sinc: true  # RawNet3 only
norm_sinc: true  # RawNet3 only
sinc_stride: 10  # RawNet3 only

## Loss - OPTIMIZED
trainfunc: aamsoftmax
margin: 0.3  # Increased
scale: 32    # Increased
nPerSpeaker: 2  # Better than 1

## Optimizer - BEST GENERALIZATION
optimizer: adamw
lr: 0.001
weight_decay: 0.0001
scheduler: cosine
min_lr: 0.00001

## Training - OPTIMIZED
batch_size: 32  # or 16 for RawNet3
max_frames: 200
eval_frames: 0
augment: true
mixedprec: true

## Data - FULL DATASET (if possible)
train_list: <full_voxceleb2_train_list>
nClasses: 5994  # Full VoxCeleb2
```

**Expected Total Improvement: 40-60% relative MinDCF reduction!** 🚀

---

## 📈 Monitoring & Evaluation

### Key Metrics to Track
1. **MinDCF** (primary metric)
2. **EER** (Equal Error Rate)
3. **Threshold** (optimal decision threshold)
4. **Training Loss** (convergence indicator)

### Validation Strategy
```yaml
test_interval: 1  # Validate every epoch (mini dataset)
test_interval: 5  # Validate every 5 epochs (full dataset)
patience: 15      # Early stopping
```

### Comparison Protocol
Always compare on the **same test set** with **same pairs**!

---

## 🔧 Practical Testing Plan

### Week 1: Quick Wins
```bash
# Experiment 1: Baseline (current)
python trainSpeakerNet_performance_updated.py \
  --config configs/mini_voxceleb1_config.yaml

# Experiment 2: ASP + log_input + better margins
# Modify config:
# encoder_type: ASP
# log_input: true
# margin: 0.3
# scale: 32
# nPerSpeaker: 2

python trainSpeakerNet_performance_updated.py \
  --config configs/mini_optimized_v1.yaml
```

### Week 2: Model Comparison
```bash
# Experiment 3: ResNetSE34V2
# config: model: ResNetSE34V2

# Experiment 4: RawNet3 (if resources available)
# config: model: RawNet3
```

### Week 3: Best Combination
Combine all best practices from above experiments.

---

## 📚 Additional Resources

### Papers to Read:
1. **RawNet3**: "Pushing the Limits of Raw Waveform Speaker Recognition"
2. **Sub-center AAMSoftmax**: "Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces"
3. **Score Normalization**: "A Study on Speaker Verification with Adaptive S-Norm"

### Codebase References:
- Your models are in: `/mnt/ricproject3/2025/Colvaiai/voxceleb_trainer/models/`
- Loss functions in: `/mnt/ricproject3/2025/Colvaiai/voxceleb_trainer/loss/`

---

## ⚠️ Important Notes

1. **Don't change everything at once!** Test one change at a time.
2. **Keep track of experiments** in your research logs.
3. **Use the same random seed** for fair comparison.
4. **Validate on multiple test sets** if possible.
5. **MinDCF depends on the cost parameters** (check dcf_p_target, dcf_c_miss, dcf_c_fa).

---

## 🎯 Expected Final Results

**Conservative Estimate (Phase 1 only):**
- Starting MinDCF: 1.0000 (your current)
- After Phase 1: 0.70-0.85 (15-30% improvement)

**Optimistic Estimate (Phase 1 + 2):**
- After Phase 1+2: 0.50-0.70 (30-50% improvement)

**Best Case (All Phases + Full Dataset):**
- After All Phases: 0.30-0.50 (50-70% improvement) 🏆

---

**Good luck with your experiments!** 🚀

Remember: **Start with Phase 1 (Quick Wins)** - these give you the best return on investment with minimal code changes!
