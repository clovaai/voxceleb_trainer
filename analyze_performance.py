#!/usr/bin/env python3
"""
Performance Analysis and Bottleneck Detection for VoxCeleb Trainer
This script analyzes:
1. Code bottlenecks
2. Memory usage patterns
3. I/O efficiency
4. GPU utilization
5. Data loading performance
"""

import os
import ast
import time
import sys
from pathlib import Path
from collections import defaultdict

class PerformanceAnalyzer:
    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self.bottlenecks = defaultdict(list)
        self.recommendations = defaultdict(list)
        
    def print_section(self, title):
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)
    
    def analyze_dataloader(self):
        """Analyze DatasetLoader.py for bottlenecks"""
        self.print_section("1. ANALYZING DATASETLOADER.PY")
        
        file_path = self.repo_path / "DatasetLoader.py"
        if not file_path.exists():
            print("  ⚠ File not found")
            return
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Check for inefficient I/O
        if 'wav.read' in content:
            issues.append("Using scipy.io.wavfile (slow)")
            self.recommendations['DataLoader'].append(
                "Replace scipy.io.wavfile with torchaudio.load for faster loading"
            )
        
        # Check for data augmentation in __getitem__
        if 'augment_wav' in content or 'add_rev' in content:
            issues.append("Data augmentation in __getitem__ (CPU bottleneck)")
            self.recommendations['DataLoader'].append(
                "Move augmentation to GPU or use torch transforms"
            )
        
        # Check for synchronous loading
        if 'num_workers' not in content:
            issues.append("No num_workers configuration mentioned")
            self.recommendations['DataLoader'].append(
                "Use num_workers > 0 for parallel data loading"
            )
        
        # Check for caching
        if 'cache' not in content.lower() and 'buffer' not in content.lower():
            issues.append("No caching mechanism detected")
            self.recommendations['DataLoader'].append(
                "Consider caching frequently used audio files in memory"
            )
        
        print(f"Found {len(issues)} potential bottlenecks:")
        for issue in issues:
            print(f"  ⚠ {issue}")
        
        self.bottlenecks['DataLoader'] = issues
    
    def analyze_training_loop(self):
        """Analyze training script for inefficiencies"""
        self.print_section("2. ANALYZING TRAINING LOOP (trainSpeakerNet.py)")
        
        file_path = self.repo_path / "trainSpeakerNet.py"
        if not file_path.exists():
            print("  ⚠ File not found")
            return
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        issues = []
        
        # Check for gradient accumulation
        if 'accumulation' not in content.lower():
            issues.append("No gradient accumulation (limits batch size)")
            self.recommendations['Training'].append(
                "Implement gradient accumulation to use larger effective batch sizes"
            )
        
        # Check for mixed precision
        if 'autocast' not in content and 'amp' not in content:
            issues.append("No mixed precision training (slower, more memory)")
            self.recommendations['Training'].append(
                "Use torch.cuda.amp for 2x speedup with FP16/BF16 training"
            )
        
        # Check for unnecessary CPU-GPU transfers
        if '.cpu()' in content or '.cuda()' in content:
            transfer_count = content.count('.cpu()') + content.count('.cuda()')
            if transfer_count > 5:
                issues.append(f"Excessive CPU-GPU transfers ({transfer_count} instances)")
                self.recommendations['Training'].append(
                    "Minimize .cpu() and .cuda() calls - keep data on GPU"
                )
        
        # Check for synchronization points
        if '.item()' in content:
            item_count = content.count('.item()')
            if item_count > 3:
                issues.append(f"Multiple .item() calls ({item_count} instances) causing synchronization")
                self.recommendations['Training'].append(
                    "Reduce .item() calls - accumulate metrics and transfer in batches"
                )
        
        # Check for distributed training
        if 'DistributedDataParallel' not in content and 'DDP' not in content:
            issues.append("No DistributedDataParallel detected")
            self.recommendations['Training'].append(
                "Use DDP for multi-GPU training (faster than DataParallel)"
            )
        
        print(f"Found {len(issues)} potential bottlenecks:")
        for issue in issues:
            print(f"  ⚠ {issue}")
        
        self.bottlenecks['Training'] = issues
    
    def analyze_model(self):
        """Analyze model architecture for inefficiencies"""
        self.print_section("3. ANALYZING MODEL ARCHITECTURE")
        
        model_files = list((self.repo_path / "models").glob("*.py"))
        
        all_issues = []
        
        for model_file in model_files:
            with open(model_file, 'r') as f:
                content = f.read()
            
            # Check for inefficient operations
            if 'for ' in content and 'forward' in content:
                # Possible Python loops in forward pass
                all_issues.append(f"{model_file.name}: Python loops in forward (use vectorization)")
                self.recommendations['Model'].append(
                    f"In {model_file.name}: Replace Python loops with vectorized operations"
                )
            
            # Check for torch.cat in loops
            if 'torch.cat' in content:
                all_issues.append(f"{model_file.name}: torch.cat usage (can be slow)")
                self.recommendations['Model'].append(
                    f"In {model_file.name}: Pre-allocate tensors instead of repeated torch.cat"
                )
            
            # Check for checkpointing
            if 'checkpoint' not in content.lower():
                self.recommendations['Model'].append(
                    f"In {model_file.name}: Consider gradient checkpointing for deep models"
                )
        
        print(f"Analyzed {len(model_files)} model files")
        print(f"Found {len(all_issues)} potential issues:")
        for issue in all_issues[:10]:
            print(f"  ⚠ {issue}")
        
        self.bottlenecks['Model'] = all_issues
    
    def analyze_io_patterns(self):
        """Analyze I/O patterns"""
        self.print_section("4. ANALYZING I/O PATTERNS")
        
        issues = []
        
        # Check config file
        config_file = self.repo_path / "configs" / "experiment_01.yaml"
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check batch size
            batch_size = config.get('batch_size', 0)
            if batch_size < 64:
                issues.append(f"Small batch size ({batch_size}) - underutilizing GPU")
                self.recommendations['I/O'].append(
                    f"Increase batch_size from {batch_size} to 128-256 for better GPU utilization"
                )
            
            # Check test interval
            test_interval = config.get('test_interval', 1)
            if test_interval == 1:
                issues.append("Testing every epoch (I/O overhead)")
                self.recommendations['I/O'].append(
                    "Increase test_interval to 2-3 to reduce testing overhead"
                )
        
        # Check for dataset on fast storage
        train_path = config.get('train_path', '')
        if train_path:
            if '/mnt/' in train_path:
                issues.append("Dataset on mounted storage (slower than local SSD)")
                self.recommendations['I/O'].append(
                    "Copy dataset to local SSD for faster I/O if possible"
                )
        
        print(f"Found {len(issues)} I/O issues:")
        for issue in issues:
            print(f"  ⚠ {issue}")
        
        self.bottlenecks['I/O'] = issues
    
    def analyze_memory_usage(self):
        """Analyze potential memory issues"""
        self.print_section("5. ANALYZING MEMORY USAGE")
        
        issues = []
        
        # Check DatasetLoader
        dl_file = self.repo_path / "DatasetLoader.py"
        if dl_file.exists():
            with open(dl_file, 'r') as f:
                content = f.read()
            
            # Check for loading entire dataset
            if 'load_all' in content.lower() or 'preload' in content.lower():
                issues.append("Potential full dataset loading (memory intensive)")
                self.recommendations['Memory'].append(
                    "Use lazy loading - load audio files on-demand"
                )
            
            # Check for memory leaks
            if 'del ' not in content:
                issues.append("No explicit tensor deletion")
                self.recommendations['Memory'].append(
                    "Add 'del' statements for large temporary tensors"
                )
        
        # Check model size
        issues.append("Large ResNet model - check if needed")
        self.recommendations['Memory'].append(
            "Consider using smaller models (ResNetSE34 vs ResNetSE152) if accuracy permits"
        )
        
        print(f"Found {len(issues)} potential memory issues:")
        for issue in issues:
            print(f"  ⚠ {issue}")
        
        self.bottlenecks['Memory'] = issues
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization recommendations"""
        self.print_section("OPTIMIZATION RECOMMENDATIONS")
        
        priorities = {
            'HIGH': [],
            'MEDIUM': [],
            'LOW': []
        }
        
        # Categorize recommendations by impact
        high_impact = [
            "Use torch.cuda.amp for 2x speedup with FP16/BF16 training",
            "Use DDP for multi-GPU training (faster than DataParallel)",
            "Use num_workers > 0 for parallel data loading",
            "Replace scipy.io.wavfile with torchaudio.load for faster loading",
        ]
        
        for category, recs in self.recommendations.items():
            for rec in recs:
                if any(h in rec for h in high_impact):
                    priorities['HIGH'].append(f"[{category}] {rec}")
                elif 'batch' in rec.lower() or 'gpu' in rec.lower():
                    priorities['MEDIUM'].append(f"[{category}] {rec}")
                else:
                    priorities['LOW'].append(f"[{category}] {rec}")
        
        print("\n🔴 HIGH PRIORITY (Implement First):")
        for i, rec in enumerate(priorities['HIGH'], 1):
            print(f"  {i}. {rec}")
        
        print("\n🟡 MEDIUM PRIORITY:")
        for i, rec in enumerate(priorities['MEDIUM'], 1):
            print(f"  {i}. {rec}")
        
        print("\n🟢 LOW PRIORITY (Nice to Have):")
        for i, rec in enumerate(priorities['LOW'][:5], 1):
            print(f"  {i}. {rec}")
    
    def generate_code_snippets(self):
        """Generate example code for key optimizations"""
        self.print_section("EXAMPLE OPTIMIZATION CODE")
        
        print("\n1. Enable Mixed Precision Training:")
        print("""
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
""")
        
        print("\n2. Optimize DataLoader:")
        print("""
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,  # Increase from 100
    num_workers=4,   # Parallel loading
    pin_memory=True, # Faster CPU->GPU transfer
    prefetch_factor=2 # Prefetch batches
)
""")
        
        print("\n3. Reduce Synchronization:")
        print("""
# BAD: Synchronizes GPU every iteration
for i, loss in enumerate(losses):
    total_loss += loss.item()  # Synchronization point!

# GOOD: Accumulate on GPU, transfer once
losses_tensor = torch.stack(losses)
total_loss = losses_tensor.sum().item()  # Single synchronization
""")
        
        print("\n4. Use torchaudio instead of scipy:")
        print("""
# BAD:
import scipy.io.wavfile as wav
fs, audio = wav.read(filename)

# GOOD:
import torchaudio
audio, fs = torchaudio.load(filename)
""")

def main():
    repo_path = '/mnt/ricproject3/2025/Colvaiai/voxceleb_trainer'
    
    print("=" * 80)
    print("VOXCELEB TRAINER PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Repository: {repo_path}\n")
    
    analyzer = PerformanceAnalyzer(repo_path)
    
    # Run all analyses
    analyzer.analyze_dataloader()
    analyzer.analyze_training_loop()
    analyzer.analyze_model()
    analyzer.analyze_io_patterns()
    analyzer.analyze_memory_usage()
    
    # Generate recommendations
    analyzer.generate_optimization_report()
    analyzer.generate_code_snippets()
    
    print("\n" + "=" * 80)
    print("Analysis complete! Review recommendations above.")
    print("=" * 80)

if __name__ == "__main__":
    main()
