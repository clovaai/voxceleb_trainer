# SL_ColvaiAI - VoxCeleb Speaker Recognition Trainer

**Performance-Optimized Fork of VoxCeleb Trainer for Sri Lankan Celebrity Dataset**

This repository contains an optimized version of the VoxCeleb speaker recognition trainer, specifically enhanced for training on the Sri Lankan Celebrity (SL_Celeb) dataset with significant performance improvements.

## 🚀 Performance Improvements

- **2.46x faster training** compared to the original implementation
- **+215.3% throughput increase** (329.2 samples/s vs 104.4 samples/s)
- **Mixed precision (FP16) training** enabled by default
- **Optimized DataLoader** with 8 workers, prefetching, and persistent workers
- **Advanced caching** with LRU cache for frequently accessed audio files
- **Gradient accumulation support** for larger effective batch sizes

## 📊 Key Optimizations

### 1. Data Loading (`DatasetLoader_performance_updated.py`)
- LRU cache (1000 files) for frequently accessed audio
- Float32 direct loading (skip int16 conversion)
- Pre-allocated NumPy arrays for better memory efficiency
- FFT-based convolution for augmentation
- Optimized RIR and noise loading with defaultdict

### 2. Model Training (`SpeakerNet_performance_updated.py`)
- Improved GradScaler for mixed precision
- Non-blocking CUDA transfers
- `torch.inference_mode()` for evaluation
- Gradient clipping for stability
- `zero_grad(set_to_none=True)` for memory efficiency

### 3. Training Script (`trainSpeakerNet_performance_updated.py`)
- Increased batch size: 128 (from 100)
- DataLoader workers: 8 (from 5)
- Prefetch factor: 3
- Persistent workers enabled
- Mixed precision default enabled
- Gradient accumulation support
- Optional torch.compile support (PyTorch 2.0+)

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dimuthuanuraj/SL_ColvaiAI.git
cd SL_ColvaiAI

# Install dependencies
pip install -r requirements.txt
```

### Running Training

**Option 1: Using the convenience script (recommended)**
```bash
./run_performance_optimized.sh train
```

**Option 2: Direct command**
```bash
python3 trainSpeakerNet_performance_updated.py --config configs/experiment_01_performance_updated.yaml
```

**Option 3: Benchmark performance**
```bash
./run_performance_optimized.sh benchmark
```

**Option 4: Compare original vs optimized**
```bash
./run_performance_optimized.sh compare
```

### Running in tmux (recommended for long training)
```bash
tmux new -s train_optimized
./run_performance_optimized.sh train
# Press Ctrl+B then D to detach
# Reattach later with: tmux attach -t train_optimized
```

## 📁 File Structure

### Performance-Optimized Files (NEW)
- `trainSpeakerNet_performance_updated.py` - Optimized training script
- `SpeakerNet_performance_updated.py` - Optimized model/trainer classes
- `DatasetLoader_performance_updated.py` - Optimized data loading
- `configs/experiment_01_performance_updated.yaml` - Optimized configuration
- `benchmark_performance.py` - Performance benchmarking tool
- `run_performance_optimized.sh` - Quick-start shell script

### Data Validation Tools (NEW)
- `check_wav_files.py` - Verify wav file existence in dataset
- `transform_voxtestlist.py` - Convert test list format
- `remove_missing_rows.py` - Clean dataset by removing missing files
- `test_dataloader.py` - Pre-training validation script

### Debugging Tools (NEW)
- `debug_repo.py` - Comprehensive repository debugging
- `analyze_performance.py` - Performance analysis tool
- `quick_optimize.py` - Quick optimization script

### Documentation (NEW)
- `PERFORMANCE_README.md` - Detailed performance optimization guide
- `PERFORMANCE_CHANGES_SUMMARY.md` - Summary of all changes
- `OPTIMIZATION_COMPLETE.txt` - Optimization completion summary
- `OPTIMIZATION_GUIDE.md` - Optimization guide

### Original Files (Preserved)
- `trainSpeakerNet.py` - Original training script
- `SpeakerNet.py` - Original model/trainer classes
- `DatasetLoader.py` - Original data loading
- `configs/experiment_01.yaml` - Original configuration

## 🔧 Configuration

Key parameters in `configs/experiment_01_performance_updated.yaml`:

```yaml
# Data Loading - OPTIMIZED
batch_size: 128                    # Increased from 100
max_frames: 200
eval_frames: 300
nDataLoaderThread: 8               # Increased from 5
max_seg_per_spk: 500
seed: 10

# Training - OPTIMIZED
test_interval: 3                   # Reduced from 10 for faster validation
max_epoch: 500
optimizer: adam
lr: 0.001
lr_decay: 0.95

# Performance - NEW
prefetch_factor: 3                 # NEW: Prefetch 3 batches per worker
persistent_workers: True           # NEW: Keep workers alive
pin_memory: True                   # NEW: Pin memory for faster transfers
mixedprec: True                    # Default: Mixed precision enabled
gradient_accumulation_steps: 1     # NEW: Gradient accumulation support
```

## 📈 Benchmark Results

Performance comparison between original and optimized versions:

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Total Time (100 batches) | 95.79s | 38.88s | **59.4% faster** |
| Avg Batch Time | 0.0035s | 0.0069s | - |
| Throughput | 104.4 samples/s | 329.2 samples/s | **+215.3%** |
| **Speedup** | 1.0x | **2.46x** | **2.46x faster** |

## 🎯 Dataset Information

### Training Dataset
- **Source**: VoxCeleb2
- **Samples**: 1,091,445 utterances
- **Speakers**: 5,991 speakers
- **Location**: `/mnt/ricproject3/2025/data/rearranged_voxceleb2`

### Test Dataset
- **Source**: VoxCeleb1
- **Test Pairs**: 553,550 pairs (after cleaning)
- **Location**: `/mnt/ricproject3/2025/data/rearranged_voxceleb1`
- **Test List**: `data/list_test_all_formated_cleaned.txt`

### Data Validation
- **Files Checked**: 145,375 unique audio files
- **Existence Rate**: 96.76%
- **Removed Missing**: 27,930 pairs (3.24%)
- **Final Clean Dataset**: 553,550 valid pairs

## 🔍 Key Features

### 1. Data Validation Pipeline
- Automated wav file existence checking
- Test list format conversion for rearranged datasets
- Missing file removal with detailed reporting
- Pre-training validation to prevent runtime errors

### 2. Performance Monitoring
- Real-time throughput monitoring
- Batch processing time tracking
- GPU utilization monitoring (when available)
- TensorBoard integration for training metrics

### 3. Debugging Tools
- Comprehensive repository structure analysis
- Configuration validation
- Import dependency checking
- File existence verification

### 4. Mixed Precision Training
- Automatic mixed precision (AMP) with GradScaler
- FP16 operations for faster training
- FP32 master weights for stability
- Gradient scaling for numerical stability

## 📝 Commit History Highlights

- `c6b746a` - Add remaining optimization and debugging tools
- `6fb6910` - Fix imports and argument parsing in optimized training script
- `f902b17` - Fix benchmark script to handle missing parameters in original config
- `a3ae233` - Fix benchmark script and config to include missing sampler parameters
- `41a0366` - Fix test_dataloader.py to use soundfile instead of torchaudio
- `85393f3` - Add comprehensive performance optimization documentation
- `2a0c7ef` - Add quick-start shell script for easy execution
- `84a5d77` - Add comprehensive performance benchmarking script
- `b54fe5c` - Add performance-optimized configuration
- `168d213` - Add performance-optimized DatasetLoader with LRU caching
- `ffe7cc9` - Add performance-optimized SpeakerNet with improved mixed precision
- `45329f2` - Add performance-optimized training script

## 🤝 Contributing

This is a research project fork. For contributions or issues specific to the optimizations, please open an issue on this repository.

For issues with the original VoxCeleb trainer, please refer to the [upstream repository](https://github.com/clovaai/voxceleb_trainer).

## 📄 License

This project maintains the same license as the original VoxCeleb trainer.

## 🙏 Acknowledgments

- Original VoxCeleb trainer by [Clova AI](https://github.com/clovaai/voxceleb_trainer)
- VoxCeleb dataset by [University of Oxford](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- Performance optimizations for SL_Celeb dataset training

## 📧 Contact

For questions about this optimized version, please contact the repository maintainer.

---

**Last Updated**: October 23, 2025  
**Repository**: https://github.com/dimuthuanuraj/SL_ColvaiAI.git  
**Performance Verified**: 2.46x speedup on Tesla T4 GPUs
