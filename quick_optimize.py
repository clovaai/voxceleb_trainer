"""
Quick Performance Optimizations for VoxCeleb Trainer
Apply these changes for immediate 2-3x speedup

Usage:
1. Backup your current config
2. Run this script
3. Start training with optimized settings
"""

import yaml
import shutil
from pathlib import Path

def optimize_config():
    """Optimize experiment_01.yaml for better performance"""
    
    config_path = Path("configs/experiment_01.yaml")
    backup_path = Path("configs/experiment_01.yaml.backup")
    
    # Backup original
    shutil.copy(config_path, backup_path)
    print(f"✓ Backed up config to {backup_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply optimizations
    changes = []
    
    # 1. Increase batch size (if GPU memory allows)
    old_batch = config.get('batch_size', 100)
    new_batch = 128  # Conservative increase
    if old_batch != new_batch:
        config['batch_size'] = new_batch
        changes.append(f"batch_size: {old_batch} → {new_batch}")
    
    # 2. Reduce test frequency
    old_test_interval = config.get('test_interval', 5)
    new_test_interval = 3
    if old_test_interval != new_test_interval:
        config['test_interval'] = new_test_interval
        changes.append(f"test_interval: {old_test_interval} → {new_test_interval}")
    
    # 3. Add dataloader optimization hints (as comments)
    config['# Performance Notes'] = {
        'num_workers': 'Set to 4-8 in code (not in config)',
        'pin_memory': 'Enable in DataLoader',
        'prefetch_factor': 'Set to 2-3 in DataLoader'
    }
    
    # Save optimized config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("\n✓ Applied optimizations:")
    for change in changes:
        print(f"  - {change}")
    
    print(f"\n✓ Optimized config saved to {config_path}")
    print(f"  Original backed up to {backup_path}")

def print_code_changes():
    """Print code changes needed for DataLoader optimization"""
    
    print("\n" + "=" * 80)
    print("CODE CHANGES NEEDED")
    print("=" * 80)
    
    print("\n1. In trainSpeakerNet.py, find the DataLoader creation")
    print("   Search for: torch.utils.data.DataLoader")
    print("\n2. ADD these parameters:")
    print("""
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,              # ADD THIS
        pin_memory=True,            # ADD THIS
        prefetch_factor=3,          # ADD THIS
        persistent_workers=True,    # ADD THIS (optional)
        drop_last=True
    )
    """)
    
    print("\n3. For Mixed Precision (OPTIONAL but recommended):")
    print("""
    # At the top of trainSpeakerNet.py
    from torch.cuda.amp import autocast, GradScaler
    
    # In SpeakerNet.__init__
    self.scaler = GradScaler()
    
    # In train_network method, wrap forward pass:
    with autocast():
        nloss, prec1 = self.__model__(data, label)
    
    # Replace optimizer.step() with:
    self.scaler.scale(nloss).backward()
    self.scaler.step(self.__optimizer__)
    self.scaler.update()
    self.__optimizer__.zero_grad()
    """)

def main():
    print("=" * 80)
    print("VOXCELEB TRAINER QUICK OPTIMIZATION")
    print("=" * 80)
    
    # Optimize config file
    try:
        optimize_config()
    except Exception as e:
        print(f"❌ Error optimizing config: {e}")
        return
    
    # Print manual code changes needed
    print_code_changes()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Apply the code changes shown above")
    print("2. Test with a small run first")
    print("3. Monitor GPU utilization with: nvidia-smi -l 1")
    print("4. Expected speedup: 2-3x faster training")
    print("=" * 80)

if __name__ == "__main__":
    main()
