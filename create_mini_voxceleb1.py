#!/usr/bin/env python3
"""
Create a mini-VoxCeleb1 dataset by randomly selecting 50 speakers.

This script:
1. Lists all speakers in the full VoxCeleb1 dataset
2. Randomly selects 50 speakers (to match proportion: 50/1211 ≈ 140/5991)
3. Creates symbolic links to preserve disk space
4. Generates a new test list file with pairs from selected speakers
"""

import os
import random
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_DIR = "/mnt/ricproject3/2025/data/rearranged_voxceleb1"
TARGET_DIR = "/mnt/ricproject3/2025/data/mini_voxceleb1"
TEST_LIST_SOURCE = "/mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt"
TEST_LIST_TARGET = "/mnt/ricproject3/2025/data/mini_test_list.txt"
TRAIN_LIST_TARGET = "/mnt/ricproject3/2025/data/mini_voxceleb1_train_list.txt"
NUM_SPEAKERS = 50  # Approximately same proportion as VoxCeleb2 mini (50/1211 ≈ 140/5991)
SEED = 42  # For reproducibility (same as VoxCeleb2 mini)

def get_all_speakers(source_dir):
    """Get list of all speaker IDs from source directory."""
    print(f"Scanning speakers in {source_dir}...")
    speakers = [d for d in os.listdir(source_dir) 
                if os.path.isdir(os.path.join(source_dir, d)) and d.startswith('id')]
    speakers.sort()
    print(f"Found {len(speakers)} speakers")
    return speakers

def select_random_speakers(speakers, num_speakers, seed):
    """Randomly select specified number of speakers."""
    random.seed(seed)
    selected = random.sample(speakers, num_speakers)
    selected.sort()
    return selected

def create_mini_dataset(source_dir, target_dir, selected_speakers, use_symlinks=True):
    """Create mini dataset by copying or symlinking speaker directories."""
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\nCreating mini dataset in {target_dir}")
    print(f"Method: {'Symbolic links' if use_symlinks else 'Copy files'}")
    
    stats = {
        'speakers': 0,
        'files': 0,
        'total_size': 0
    }
    
    for speaker_id in tqdm(selected_speakers, desc="Processing speakers"):
        source_speaker = os.path.join(source_dir, speaker_id)
        target_speaker = os.path.join(target_dir, speaker_id)
        
        if use_symlinks:
            # Create symbolic link to entire speaker directory
            if os.path.exists(target_speaker):
                os.remove(target_speaker)
            os.symlink(source_speaker, target_speaker)
            stats['speakers'] += 1
            
            # Count files
            files = list(Path(source_speaker).glob('*.wav'))
            stats['files'] += len(files)
            stats['total_size'] += sum(f.stat().st_size for f in files)
        else:
            # Copy entire directory
            if os.path.exists(target_speaker):
                shutil.rmtree(target_speaker)
            shutil.copytree(source_speaker, target_speaker)
            stats['speakers'] += 1
            
            files = list(Path(target_speaker).glob('*.wav'))
            stats['files'] += len(files)
            stats['total_size'] += sum(f.stat().st_size for f in files)
    
    return stats

def create_mini_test_list(source_list, target_list, selected_speakers, balance_ratio=0.5):
    """Create BALANCED test list file for mini dataset with only pairs from selected speakers.
    
    Args:
        source_list: Source test list file
        target_list: Target mini test list file
        selected_speakers: Set of selected speaker IDs
        balance_ratio: Desired ratio of positive pairs (default 0.5 for 50/50 balance)
    """
    
    print(f"\nCreating BALANCED mini test list (target: {balance_ratio*100:.0f}% positive pairs)...")
    selected_set = set(selected_speakers)
    
    # First pass: collect all valid pairs
    positive_pairs = []  # label=1 (same speaker)
    negative_pairs = []  # label=0 (different speakers)
    lines_total = 0
    
    with open(source_list, 'r') as f_in:
        for line in f_in:
            lines_total += 1
            line = line.strip()
            if not line:
                continue
            
            # Extract speaker IDs from test pair
            # Format: 1 id10001/00001_Y8hIVOBuels.wav id10001/00001_1zcIwhmdeo4.wav
            parts = line.split()
            if len(parts) >= 3:
                label = parts[0]
                file1_path = parts[1]
                file2_path = parts[2]
                
                # Extract speaker IDs from file paths
                speaker1 = file1_path.split('/')[0]
                speaker2 = file2_path.split('/')[0]
                
                # Include pair only if both speakers are in selected set
                if speaker1 in selected_set and speaker2 in selected_set:
                    if label == '1':
                        positive_pairs.append(line)
                    else:
                        negative_pairs.append(line)
    
    print(f"Collected pairs from selected speakers:")
    print(f"  Positive pairs (label=1): {len(positive_pairs)}")
    print(f"  Negative pairs (label=0): {len(negative_pairs)}")
    
    # Balance the dataset
    if len(positive_pairs) == 0 or len(negative_pairs) == 0:
        print("⚠️  Warning: Cannot balance - one class has no pairs!")
        # Write whatever we have
        with open(target_list, 'w') as f_out:
            for line in positive_pairs:
                f_out.write(line + '\n')
            for line in negative_pairs:
                f_out.write(line + '\n')
        return len(positive_pairs) + len(negative_pairs), len(positive_pairs), len(negative_pairs)
    
    # Calculate how many pairs to use from each class
    total_desired = min(len(positive_pairs), len(negative_pairs)) * 2
    num_positive = int(total_desired * balance_ratio)
    num_negative = total_desired - num_positive
    
    # Adjust if we don't have enough pairs
    if num_positive > len(positive_pairs):
        num_positive = len(positive_pairs)
        num_negative = int(num_positive * (1 - balance_ratio) / balance_ratio)
    if num_negative > len(negative_pairs):
        num_negative = len(negative_pairs)
        num_positive = int(num_negative * balance_ratio / (1 - balance_ratio))
    
    # Randomly sample pairs to maintain balance
    random.seed(SEED)
    selected_positive = random.sample(positive_pairs, num_positive)
    selected_negative = random.sample(negative_pairs, num_negative)
    
    # Shuffle and write
    all_pairs = selected_positive + selected_negative
    random.shuffle(all_pairs)
    
    with open(target_list, 'w') as f_out:
        for line in all_pairs:
            f_out.write(line + '\n')
    
    lines_written = len(all_pairs)
    actual_balance = num_positive / lines_written * 100 if lines_written > 0 else 0
    
    print(f"\nBalanced test list created:")
    print(f"  Total pairs: {lines_written}")
    print(f"  Positive pairs (label=1): {num_positive} ({num_positive/lines_written*100:.1f}%)")
    print(f"  Negative pairs (label=0): {num_negative} ({num_negative/lines_written*100:.1f}%)")
    print(f"  ✅ Balance achieved: {actual_balance:.1f}% positive pairs")
    
    return lines_written, num_positive, num_negative

def save_speaker_list(speakers, filepath):
    """Save list of selected speakers to file."""
    with open(filepath, 'w') as f:
        for speaker in speakers:
            f.write(speaker + '\n')
    print(f"Saved speaker list to {filepath}")

def create_train_list_from_folders(target_dir, train_list_file, selected_speakers):
    """Create training list from the audio files in mini dataset folders."""
    
    print(f"\nCreating training list from mini dataset folders...")
    
    lines_written = 0
    
    with open(train_list_file, 'w') as f_out:
        for speaker_id in tqdm(selected_speakers, desc="Creating train list"):
            speaker_path = os.path.join(target_dir, speaker_id)
            
            # Get all wav files for this speaker
            wav_files = sorted(Path(speaker_path).glob('*.wav'))
            
            for wav_file in wav_files:
                # Format: speaker_id /full/path/to/file.wav
                line = f"{speaker_id} {wav_file}\n"
                f_out.write(line)
                lines_written += 1
    
    print(f"Created training list with {lines_written} entries")
    return lines_written

def main():
    print("=" * 60)
    print("Mini-VoxCeleb1 Dataset Creator")
    print("=" * 60)
    
    # Get all speakers
    all_speakers = get_all_speakers(SOURCE_DIR)
    
    if len(all_speakers) < NUM_SPEAKERS:
        print(f"Error: Only {len(all_speakers)} speakers available, cannot select {NUM_SPEAKERS}")
        return
    
    # Select random speakers
    print(f"\nRandomly selecting {NUM_SPEAKERS} speakers (seed={SEED})...")
    selected_speakers = select_random_speakers(all_speakers, NUM_SPEAKERS, SEED)
    
    print(f"Selected speakers: {selected_speakers[:5]}...{selected_speakers[-5:]}")
    
    # Save speaker list
    speaker_list_file = os.path.join(os.path.dirname(TARGET_DIR), "mini_voxceleb1_speakers.txt")
    save_speaker_list(selected_speakers, speaker_list_file)
    
    # Create mini dataset
    stats = create_mini_dataset(SOURCE_DIR, TARGET_DIR, selected_speakers, use_symlinks=True)
    
    print("\n" + "=" * 60)
    print("Mini Dataset Statistics:")
    print("=" * 60)
    print(f"Speakers: {stats['speakers']}")
    print(f"Audio files: {stats['files']}")
    print(f"Total size: {stats['total_size'] / (1024**3):.2f} GB")
    print(f"Avg files per speaker: {stats['files'] / stats['speakers']:.1f}")
    
    # Create mini test list
    if os.path.exists(TEST_LIST_SOURCE):
        total_pairs, same_pairs, diff_pairs = create_mini_test_list(
            TEST_LIST_SOURCE, TEST_LIST_TARGET, selected_speakers
        )
        print(f"\nTest pairs in mini dataset: {total_pairs}")
        print(f"  Positive pairs (same speaker): {same_pairs}")
        print(f"  Negative pairs (different speaker): {diff_pairs}")
        
        if total_pairs > 0:
            balance = same_pairs / total_pairs * 100
            print(f"  Balance: {balance:.1f}% positive pairs")
    else:
        print(f"\nWarning: Test list not found at {TEST_LIST_SOURCE}")
    
    # Create training list from the mini dataset folders
    train_files = create_train_list_from_folders(TARGET_DIR, TRAIN_LIST_TARGET, selected_speakers)
    
    print("\n" + "=" * 60)
    print("Mini-VoxCeleb1 Creation Complete!")
    print("=" * 60)
    print(f"\nDataset location: {TARGET_DIR}")
    print(f"Test list: {TEST_LIST_TARGET}")
    print(f"Train list: {TRAIN_LIST_TARGET}")
    print(f"Speaker list: {speaker_list_file}")
    
    print("\nTo use this mini dataset for TRAINING:")
    print(f"  train_path: {TARGET_DIR}")
    print(f"  train_list: {TRAIN_LIST_TARGET}")
    print(f"  nClasses: {NUM_SPEAKERS}")
    
    print("\nTo use this mini dataset for TESTING:")
    print(f"  test_path: {TARGET_DIR}")
    print(f"  test_list: {TEST_LIST_TARGET}")
    
    print("\n⚠️  Note: This mini test set is for development only.")
    print("   For final evaluation, use the full VoxCeleb1 test set.")

if __name__ == "__main__":
    main()
