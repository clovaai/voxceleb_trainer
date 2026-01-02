#!/usr/bin/env python3
"""
Corrupted Audio File Scanner for VoxCeleb Dataset
==================================================

Purpose:
--------
Identifies corrupted or unreadable audio files in the VoxCeleb dataset that
cause LibsndfileError during training. This tool scans all audio files and
attempts to load them with soundfile library to verify integrity.

Use Cases:
----------
1. Pre-training validation: Check dataset before starting long training runs
2. Error debugging: Identify specific corrupted file causing training crashes
3. Dataset cleaning: Generate list of files to exclude or re-download

Usage:
------
    python3 check_corrupted_audio.py

Output:
-------
- Prints progress every 1000 files
- Prints corrupted files immediately when found
- Saves list to: corrupted_audio_files.txt

Author: Anuraj
Date: December 31, 2025
Project: MLP-Mixer Speaker Verification with Knowledge Distillation

Notes:
------
- Supports .wav, .flac, .m4a, .aac extensions
- Scans recursively through directory tree
- Handles exceptions gracefully (file not found, permission denied, etc.)
- For full VoxCeleb2: ~1.09M files, takes ~30-60 minutes to scan
"""

import os
import soundfile as sf

def check_audio_file(filepath):
    """Check if an audio file can be loaded"""
    try:
        data, samplerate = sf.read(filepath)
        return True, None
    except Exception as e:
        return False, str(e)

def scan_directory(directory, extensions=['.wav', '.flac', '.m4a', '.aac']):
    """Scan directory for audio files and check each one"""
    print(f"Scanning directory: {directory}")
    
    # Collect all audio files
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Check each file
    corrupted_files = []
    total = len(audio_files)
    for idx, filepath in enumerate(audio_files):
        if (idx + 1) % 1000 == 0:
            print(f"Progress: {idx + 1}/{total} files checked ({100*(idx+1)/total:.1f}%)")
        
        is_valid, error = check_audio_file(filepath)
        if not is_valid:
            corrupted_files.append((filepath, error))
            print(f"\n❌ CORRUPTED: {filepath}")
            print(f"   Error: {error}")
    
    return corrupted_files

if __name__ == "__main__":
    # Check VoxCeleb2 training data
    train_path = "/mnt/ricproject2/voxceleb_new/voxceleb2"
    
    print("=" * 80)
    print("AUDIO FILE INTEGRITY CHECKER")
    print("=" * 80)
    
    corrupted = scan_directory(train_path)
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Found {len(corrupted)} corrupted file(s)")
    print("=" * 80)
    
    if corrupted:
        print("\nCorrupted files:")
        for filepath, error in corrupted:
            print(f"  - {filepath}")
            print(f"    Error: {error}")
        
        # Save to file
        output_file = "corrupted_audio_files.txt"
        with open(output_file, 'w') as f:
            for filepath, error in corrupted:
                f.write(f"{filepath}\n")
        print(f"\n✓ Corrupted file list saved to: {output_file}")
    else:
        print("\n✓ All audio files are valid!")
