#!/usr/bin/env python3
"""
Create a mini-VoxCeleb2 dataset by randomly selecting 140 speakers.

This script:
1. Lists all speakers in the full VoxCeleb2 dataset
2. Randomly selects 140 speakers
3. Creates symbolic links to preserve disk space
4. Generates a new training list file
"""

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_DIR = "/mnt/ricproject3/2025/data/rearranged_voxceleb2"
TARGET_DIR = "/mnt/ricproject3/2025/data/mini_voxceleb2"
TRAIN_LIST_SOURCE = "/mnt/ricproject3/2025/data/train_list.txt"
TRAIN_LIST_TARGET = "/mnt/ricproject3/2025/data/mini_train_list.txt"
NUM_SPEAKERS = 140
SEED = 42  # For reproducibility

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

def create_mini_train_list(source_list, target_list, selected_speakers):
    """Create training list file for mini dataset."""
    
    print(f"\nCreating mini training list...")
    selected_set = set(selected_speakers)
    
    lines_written = 0
    lines_total = 0
    
    with open(source_list, 'r') as f_in, open(target_list, 'w') as f_out:
        for line in f_in:
            lines_total += 1
            line = line.strip()
            if not line:
                continue
            
            # Extract speaker ID from line
            # Format: id00012 /path/to/file.wav
            parts = line.split()
            if len(parts) >= 2:
                speaker_id = parts[0]  # First column is the speaker ID
                
                if speaker_id in selected_set:
                    f_out.write(line + '\n')
                    lines_written += 1
    
    print(f"Processed {lines_total} lines from source")
    print(f"Wrote {lines_written} lines to mini train list")
    
    return lines_written

def save_speaker_list(speakers, filepath):
    """Save list of selected speakers to file."""
    with open(filepath, 'w') as f:
        for speaker in speakers:
            f.write(speaker + '\n')
    print(f"Saved speaker list to {filepath}")

def main():
    print("=" * 60)
    print("Mini-VoxCeleb2 Dataset Creator")
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
    speaker_list_file = os.path.join(os.path.dirname(TARGET_DIR), "mini_voxceleb2_speakers.txt")
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
    
    # Create mini training list
    if os.path.exists(TRAIN_LIST_SOURCE):
        lines = create_mini_train_list(TRAIN_LIST_SOURCE, TRAIN_LIST_TARGET, selected_speakers)
        print(f"\nTraining samples in mini dataset: {lines}")
    else:
        print(f"\nWarning: Training list not found at {TRAIN_LIST_SOURCE}")
    
    print("\n" + "=" * 60)
    print("Mini-VoxCeleb2 Creation Complete!")
    print("=" * 60)
    print(f"\nDataset location: {TARGET_DIR}")
    print(f"Train list: {TRAIN_LIST_TARGET}")
    print(f"Speaker list: {speaker_list_file}")
    
    print("\nTo use this mini dataset, update your config file:")
    print(f"  train_path: {TARGET_DIR}")
    print(f"  train_list: {TRAIN_LIST_TARGET}")

if __name__ == "__main__":
    main()
