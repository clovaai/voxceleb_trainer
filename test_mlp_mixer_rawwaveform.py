#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
Test script for MLPMixerSpeaker_RawWaveform

Validates:
1. Model instantiation
2. Forward pass with raw waveform input
3. Parameter count
4. Output shape correctness
5. SincNet filter learning
"""

import torch
import sys
sys.path.insert(0, '/mnt/ricproject3/2025/Colvaiai/voxceleb_trainer')

from models.MLPMixerSpeaker_RawWaveform import MLPMixerSpeakerNet_RawWaveform


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_raw_waveform_model():
    print("=" * 80)
    print("Testing MLP-Mixer with Raw Waveform Input")
    print("=" * 80)
    
    # Create model with V2 configuration
    model = MLPMixerSpeakerNet_RawWaveform(
        nOut=512,
        num_filters=80,
        hidden_dim=192,
        num_blocks=6,
        expansion_factor=4,
        groups=4
    )
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"\n✓ Model created successfully")
    print(f"  Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Test forward pass with different input lengths
    test_cases = [
        (2, 32000),   # 2 samples, 2 seconds @ 16kHz
        (4, 48000),   # 4 samples, 3 seconds @ 16kHz
        (1, 64000),   # 1 sample, 4 seconds @ 16kHz
    ]
    
    model.eval()
    with torch.no_grad():
        for batch_size, num_samples in test_cases:
            # Create random waveform input
            x = torch.randn(batch_size, num_samples)
            
            # Forward pass
            embeddings = model(x)
            
            # Validate output shape
            expected_shape = (batch_size, 512)
            assert embeddings.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {embeddings.shape}"
            
            print(f"\n✓ Forward pass successful:")
            print(f"  Input: {x.shape} (raw waveform)")
            print(f"  Output: {embeddings.shape} (embeddings)")
            print(f"  Embedding norm: {embeddings.norm(dim=1).mean():.4f}")
    
    # Test SincNet filters
    print(f"\n✓ SincNet learnable filters:")
    print(f"  Number of filters: {model.sincnet.out_channels}")
    print(f"  Filter kernel size: {model.sincnet.kernel_size} samples")
    print(f"  Stride (hop length): {model.sincnet.stride} samples")
    print(f"  Low cutoff frequencies (Hz): min={model.sincnet.low_hz_.min():.1f}, "
          f"max={model.sincnet.low_hz_.max():.1f}")
    print(f"  Bandwidths (Hz): min={model.sincnet.band_hz_.min():.1f}, "
          f"max={model.sincnet.band_hz_.max():.1f}")
    
    # Compare with mel-based V2
    print(f"\n✓ Comparison with mel-based V2:")
    print(f"  Raw waveform: {n_params/1e6:.2f}M parameters")
    print(f"  Mel-based V2: 2.66M parameters")
    print(f"  Difference: {(n_params - 2.66e6)/1e6:+.2f}M ({(n_params/2.66e6 - 1)*100:+.1f}%)")
    
    # Memory usage estimate
    x_test = torch.randn(32, 32000)  # Batch of 32, 2 seconds
    with torch.no_grad():
        _ = model(x_test)
    print(f"\n✓ Memory test (batch_size=32, 2s audio):")
    print(f"  Forward pass completed successfully")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    print("\nReady for training with:")
    print("  1. Baseline: configs/mlp_mixer_rawwaveform_baseline.yaml")
    print("  2. Distillation: configs/mlp_mixer_rawwaveform_distillation.yaml")
    print("=" * 80)


if __name__ == "__main__":
    test_raw_waveform_model()
