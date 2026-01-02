#!/usr/bin/env python3
"""
Script to test if the test list is working correctly before running full training.
This will verify:
1. Test list file can be loaded
2. All wav files in the test list exist
3. Data can be loaded without errors
"""

import os
import sys
import torch
import soundfile  # Use soundfile instead of torchaudio for better compatibility

def test_file_paths(test_list, test_path):
    """Test if all files in the test list exist"""
    print("=" * 80)
    print("TESTING FILE PATHS")
    print("=" * 80)
    print(f"Test list: {test_list}")
    print(f"Test path: {test_path}")
    
    if not os.path.exists(test_list):
        print(f"❌ ERROR: Test list file not found: {test_list}")
        return False
    
    missing_files = []
    total_pairs = 0
    
    with open(test_list, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            total_pairs += 1
            label, wav1, wav2 = parts[0], parts[1], parts[2]
            
            path1 = os.path.join(test_path, wav1)
            path2 = os.path.join(test_path, wav2)
            
            if not os.path.exists(path1):
                missing_files.append((line_num, wav1))
            if not os.path.exists(path2):
                missing_files.append((line_num, wav2))
            
            # Print progress
            if line_num % 50000 == 0:
                print(f"  Checked {line_num} lines...")
    
    print(f"\n✓ Total pairs checked: {total_pairs}")
    
    if missing_files:
        print(f"\n❌ Found {len(missing_files)} missing files:")
        for line_num, file_path in missing_files[:10]:
            print(f"  Line {line_num}: {file_path}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        return False
    else:
        print("✓ All files exist!")
        return True

def test_audio_loading(test_list, test_path, num_samples=10):
    """Test if audio files can be loaded"""
    print("\n" + "=" * 80)
    print("TESTING AUDIO LOADING")
    print("=" * 80)
    
    samples = []
    with open(test_list, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                samples.append((parts[1], parts[2]))
    
    print(f"Testing {len(samples)} sample pairs...\n")
    
    for idx, (wav1, wav2) in enumerate(samples, 1):
        try:
            path1 = os.path.join(test_path, wav1)
            path2 = os.path.join(test_path, wav2)
            
            # Try to load audio using soundfile (more compatible)
            audio1, sr1 = soundfile.read(path1)
            audio2, sr2 = soundfile.read(path2)
            
            print(f"  Sample {idx}:")
            print(f"    ✓ {wav1} - Shape: {audio1.shape}, SR: {sr1}")
            print(f"    ✓ {wav2} - Shape: {audio2.shape}, SR: {sr2}")
            
        except Exception as e:
            print(f"  Sample {idx}:")
            print(f"    ❌ Error loading audio: {e}")
            return False
    
    print("\n✓ All samples loaded successfully!")
    return True

def main():
    # Configuration from experiment_01.yaml
    test_list = '/mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt'
    test_path = '/mnt/ricproject3/2025/data/rearranged_voxceleb1'
    
    print("Testing VoxCeleb Test List Configuration")
    print("=" * 80)
    
    # Test 1: Check if all files exist
    success1 = test_file_paths(test_list, test_path)
    
    # Test 2: Try loading some audio files
    success2 = test_audio_loading(test_list, test_path, num_samples=10)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    if success1 and success2:
        print("✅ All tests passed! The test list is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
