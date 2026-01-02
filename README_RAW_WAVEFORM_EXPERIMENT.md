# MLP-Mixer with Raw Waveform Input - Experiment P3

**Date:** December 31, 2025  
**Status:** Implementation complete, ready for training  
**Hypothesis:** Learnable SincNet filters may capture speaker-discriminative features automatically, potentially matching or outperforming fixed mel-spectrogram preprocessing

---

## Overview

This experiment tests MLP-Mixer with **raw waveform input** instead of mel-spectrogram features. Inspired by SincNet (Ravanelli & Bengio, 2018), we replace fixed mel-filterbanks with learnable bandpass filters that can adapt to the speaker verification task.

### Key Innovation

```
Traditional MLP-Mixer (V2):
  Raw Audio → Mel-Spectrogram (fixed) → CNN → MLP-Mixer → Embedding

Raw Waveform MLP-Mixer (P3):
  Raw Audio → SincNet (learnable) → CNN → MLP-Mixer → Embedding
              ↑
              80 learnable bandpass filters (replace fixed mel-filterbanks)
```

---

## Architecture Details

### SincNet Frontend (Learnable Filters)

- **80 learnable bandpass filters** (same count as mel-filterbanks)
- **Kernel size:** 251 samples (~15ms @ 16kHz)
- **Hop length:** 160 samples (10ms @ 16kHz, same as mel-spec)
- **Learnable parameters:**
  - Low cutoff frequencies (initialized 30 Hz - 7.6 kHz)
  - Bandwidths (initialized 23 Hz - 261 Hz)
- **Initialization:** Mel-scale spacing (mimics mel-filterbanks initially)

### Additional Feature Extraction

After SincNet, additional CNN layers process learned features:
```python
Conv1d(80, 80, k=5) → LeakyReLU → MaxPool(3)
Conv1d(80, 80, k=5) → LeakyReLU → MaxPool(3)
```

This matches the original SincNet architecture for temporal feature extraction.

### MLP-Mixer Encoder

Same as V2:
- **6 MLP-Mixer blocks**
- **Hidden dimension:** 192
- **ID Conv + Token Mixing + MFM + Channel Mixing**
- **Attentive Statistics Pooling**
- **512-dimensional embeddings**

---

## Model Comparison

| Model | Input | Frontend | Parameters | Inference Speed |
|-------|-------|----------|-----------|----------------|
| **V2** | Mel-spec | Fixed filters | 2.66M | 292 samples/sec |
| **P3** | Raw wave | Learnable filters | 3.48M | ~280 samples/sec (est.) |
| **Difference** | - | +0.82M | +30.9% | -4% (est.) |

**Parameter increase:** Primarily from SincNet learnable filters and additional CNN layers.

---

## Research Questions

1. **Performance:** Can raw waveform + distillation match mel-based V2 (10.32% EER)?
2. **Feature Learning:** Do learned filters discover better speaker-discriminative bands?
3. **Robustness:** Is raw waveform more robust to noisy/distorted audio?
4. **Training Time:** How much slower is training with raw input?

---

## Experimental Setup

### Two-Phase Approach

**Phase 1: Baseline (No Distillation)**
- Purpose: Validate raw waveform processing
- Expected EER: 12-14%
- Training: 50 epochs, mini dataset
- Config: `configs/mlp_mixer_rawwaveform_baseline.yaml`

**Phase 2: Distillation**
- Purpose: Test raw waveform + teacher knowledge
- Expected EER: 10.5-11.5% (vs mel-based V2: 10.32%)
- Training: 100 epochs, mini dataset
- Config: `configs/mlp_mixer_rawwaveform_distillation.yaml`
- Teacher: LSTM+Autoencoder (9.68% EER)
- Distillation: α=0.7, T=4.0, Cosine loss

### Dataset

**Training:** Mini VoxCeleb2 (30K samples)
- Faster iteration for initial testing
- No data augmentation (for stability)

**Validation:** VoxCeleb1-O test set
- Standard evaluation protocol

---

## Files Created

### Core Implementation

```
models/MLPMixerSpeaker_RawWaveform.py (389 lines)
├── SincConv_fast (learnable bandpass filters)
├── MLPMixerBlock (same as V2)
├── AttentiveStatsPooling (same as V2)
└── MLPMixerSpeakerNet_RawWaveform (main model)
```

### Configuration Files

```
configs/mlp_mixer_rawwaveform_baseline.yaml
├── 50 epochs, no distillation
└── Expected EER: 12-14%

configs/mlp_mixer_rawwaveform_distillation.yaml
├── 100 epochs, distillation (α=0.7)
└── Expected EER: 10.5-11.5%
```

### Training Scripts

```
train_mlp_mixer_rawwaveform_baseline.sh
└── Phase 1 training (baseline)

train_mlp_mixer_rawwaveform_distillation.sh
└── Phase 2 training (distillation)
```

### Testing

```
test_mlp_mixer_rawwaveform.py
└── Validates implementation (all tests passed ✓)
```

---

## How to Run

### 1. Test Implementation

```bash
python3 test_mlp_mixer_rawwaveform.py
```

**Expected output:**
```
✓ Model created successfully (3.48M parameters)
✓ Forward pass successful
✓ SincNet learnable filters initialized
✓ All tests passed!
```

### 2. Phase 1: Baseline Training

```bash
bash train_mlp_mixer_rawwaveform_baseline.sh
```

**Monitor progress:**
```bash
tail -f exps/mlp_mixer_rawwaveform_baseline/result/scores.txt
```

**Success criteria:** EER < 14% (validates raw waveform processing)

### 3. Phase 2: Distillation Training

```bash
bash train_mlp_mixer_rawwaveform_distillation.sh
```

**Monitor progress:**
```bash
tail -f exps/mlp_mixer_rawwaveform_distillation/result/scores.txt
```

**Success criteria:**
- **Excellent:** EER ≤ 10.5% (matches/beats mel-based V2)
- **Good:** EER 10.5-11.0% (competitive)
- **Acceptable:** EER 11.0-11.5% (slightly worse)
- **Poor:** EER > 11.5% (mel preprocessing superior)

---

## Expected Results

### Scenario 1: Raw Waveform Wins (EER ≤ 10.5%)

**Implications:**
- Learnable filters capture better features than fixed mel
- Replace mel preprocessing in production models
- Potential for better robustness on noisy audio

**Next steps:**
- Full VoxCeleb2 training (expect 8.5-9.0% EER)
- Analyze learned filter frequencies (what did it discover?)
- Test on noisy evaluation sets

### Scenario 2: Comparable Performance (EER 10.5-11.5%)

**Implications:**
- Raw waveform viable alternative to mel
- Choice depends on use case (flexibility vs speed)
- May excel in specific conditions (noise, distortion)

**Next steps:**
- Robustness testing (noise, codec distortion)
- Computational cost analysis
- Domain-specific evaluation

### Scenario 3: Mel Preprocessing Wins (EER > 11.5%)

**Implications:**
- Fixed mel-filterbanks better for speaker verification
- Additional parameters not worth the cost
- Stick with mel-based V2 for production

**Next steps:**
- Document why mel is better (spectral properties)
- Archive raw waveform experiment
- Focus on other improvements (P4: hybrid architecture)

---

## Technical Details

### SincNet Filter Equations

Bandpass filter impulse response:
```
h[n] = (sin(2π f_high n) - sin(2π f_low n)) / (πn)
```

Where:
- `f_low`: Learnable low cutoff frequency
- `f_high = f_low + bandwidth`: High cutoff frequency
- `bandwidth`: Learnable parameter

Initialized with mel-scale spacing:
```
f_mel[i] = 700 * (10^(mel[i]/2595) - 1)
```

### Memory Requirements

Raw waveform requires more memory during training:
- **Input size:** [batch, samples] (e.g., [32, 32000])
- **After SincNet:** [32, 80, time_frames]
- **Mel-based:** [32, 80, time_frames] (smaller initial memory)

**Batch size reduced:** 32 (vs 64 for mel-based) to fit in GPU memory

---

## Code Impact Analysis

### Zero Impact on Existing Code ✓

**New files only:**
- `models/MLPMixerSpeaker_RawWaveform.py` (new)
- `configs/mlp_mixer_rawwaveform_*.yaml` (new)
- `train_mlp_mixer_rawwaveform_*.sh` (new)
- `test_mlp_mixer_rawwaveform.py` (new)

**No modifications to:**
- Existing model files
- Training scripts
- Evaluation code
- Other configurations

**Compatibility:**
- Uses same training framework (trainSpeakerNet.py, trainSpeakerNet_distillation.py)
- Same distillation wrapper (DistillationWrapper.py)
- Same evaluation protocol

---

## References

1. **SincNet:** Ravanelli, M., & Bengio, Y. (2018). "Speaker Recognition from Raw Waveform with SincNet." IEEE SLT 2018.

2. **MLP-Mixer Paper:** Anonymous authors. "A Speaker Verification System Based on a Modified MLP-Mixer Student Model for Transformer Compression."

3. **Our Previous Work:**
   - V1: 16.13% EER (MSE loss bug)
   - V2: 10.32% EER (Cosine loss fix) ← **Baseline to beat**
   - V2_Large: 14.84% EER (α too high)
   - V2_Large_lowAlpha: 10.11% EER (α=0.4 optimal)

---

## Timeline

- **December 31, 2025:** Implementation complete, testing passed
- **Next:** Baseline training (Phase 1, 4-6 hours)
- **Next:** Distillation training (Phase 2, 12-16 hours)
- **Next:** Results analysis and documentation

---

## Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Baseline EER** | < 14% | Validates raw waveform processing |
| **Distillation EER** | < 11.5% | Competitive with mel-based V2 |
| **Training time** | < 2× mel-based | Acceptable computational cost |
| **Inference speed** | > 250 samples/sec | Production viable |

---

## Contact & Documentation

**Research logs:** `research_logs/2025-12-30-31-experimental-results-analysis.md`  
**Git commits:** All changes will be committed separately (zero impact on existing code)

---

**Status:** ✓ Ready for training  
**Next action:** Run Phase 1 baseline training
