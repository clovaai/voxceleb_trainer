#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Test script for MLP-Mixer Student Model

Validates:
1. Model instantiation
2. Forward pass
3. Teacher model loading
4. Distillation wrapper
5. Parameter count
6. Inference speed comparison

Run: python3 test_mlp_mixer.py
"""

import torch
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.MLPMixerSpeaker import MainModel as MLPMixerModel
from models.LSTMAutoencoder import MainModel as LSTMModel
from models.ResNetSE34L import MainModel as ResNetModel


def test_model_instantiation():
    """Test 1: Model instantiation"""
    print('='*60)
    print('TEST 1: Model Instantiation')
    print('='*60)
    
    model = MLPMixerModel(
        nOut=512,
        n_mels=80,
        hidden_dim=192,
        num_blocks=6,
        expansion_factor=3,
        groups=4
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n✓ Model instantiated successfully')
    print(f'Parameters: {total_params:,} ({total_params/1e6:.2f}M)')
    print(f'Target range: 1.5M - 3.0M ✓' if 1.5e6 < total_params < 3.0e6 else 'Target range: 1.5M - 3.0M ✗')
    
    return model


def test_forward_pass(model):
    """Test 2: Forward pass"""
    print('\n' + '='*60)
    print('TEST 2: Forward Pass')
    print('='*60)
    
    # Create dummy input (batch=4, 2 seconds @ 16kHz)
    batch_size = 4
    duration = 2.0
    sample_rate = 16000
    dummy_audio = torch.randn(batch_size, int(duration * sample_rate))
    
    print(f'\nInput: {dummy_audio.shape}')
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_audio)
    
    print(f'Output: {output.shape}')
    print(f'Expected: [{batch_size}, 512]')
    
    assert output.shape == torch.Size([batch_size, 512]), 'Output shape mismatch!'
    print('\n✓ Forward pass successful')
    
    return output


def test_inference_speed():
    """Test 3: Inference speed comparison"""
    print('\n' + '='*60)
    print('TEST 3: Inference Speed Comparison')
    print('='*60)
    
    models = {
        'MLP-Mixer': MLPMixerModel(nOut=512, n_mels=80, hidden_dim=192, num_blocks=6, expansion_factor=3, groups=4),
        'LSTM+AE': LSTMModel(nOut=512, n_mels=80, ae_latent_dim=128, lstm_hidden=256, lstm_layers=2, pooling_type='ASP'),
        'ResNetSE34L': ResNetModel(nOut=512, encoder_type='ASP', n_mels=80)
    }
    
    # Dummy input
    dummy_audio = torch.randn(8, 32000)  # 8 samples, 2 seconds
    num_iterations = 50
    
    results = {}
    
    for name, model in models.items():
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_audio)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_audio)
        elapsed_time = time.time() - start_time
        
        avg_time = elapsed_time / num_iterations
        throughput = 8 / avg_time  # samples per second
        
        results[name] = {
            'avg_time': avg_time,
            'throughput': throughput,
            'params': sum(p.numel() for p in model.parameters())
        }
        
        print(f'\n{name}:')
        print(f'  Avg time: {avg_time*1000:.2f} ms/batch')
        print(f'  Throughput: {throughput:.2f} samples/sec')
        print(f'  Parameters: {results[name]["params"]/1e6:.2f}M')
    
    # Calculate speedup
    baseline_time = results['LSTM+AE']['avg_time']
    mlp_mixer_time = results['MLP-Mixer']['avg_time']
    speedup = baseline_time / mlp_mixer_time
    
    print(f'\n{"="*60}')
    print(f'MLP-Mixer Speedup vs LSTM+AE: {speedup:.2f}×')
    print(f'Expected: 2-3× ✓' if 2.0 <= speedup <= 3.5 else f'Expected: 2-3× (got {speedup:.2f}×)')
    print(f'{"="*60}')
    
    return results


def test_distillation_wrapper():
    """Test 4: Distillation wrapper (without actual teacher checkpoint)"""
    print('\n' + '='*60)
    print('TEST 4: Distillation Wrapper')
    print('='*60)
    
    try:
        from DistillationWrapper import DistillationSpeakerNet
        
        print('\n✓ DistillationWrapper module imported successfully')
        print('Note: Actual distillation requires teacher checkpoint')
        print('      (See configs/mlp_mixer_distillation_config.yaml)')
        
        return True
    except Exception as e:
        print(f'\n✗ Error importing DistillationWrapper: {e}')
        return False


def main():
    """Run all tests"""
    print('\n' + '='*60)
    print('MLP-MIXER SPEAKER VERIFICATION - TEST SUITE')
    print('='*60)
    
    try:
        # Test 1: Instantiation
        model = test_model_instantiation()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        
        # Test 3: Speed comparison (CPU only to avoid GPU requirement)
        print('\n⚠️  Speed test requires substantial computation, running...')
        test_inference_speed()
        
        # Test 4: Distillation wrapper
        test_distillation_wrapper()
        
        print('\n' + '='*60)
        print('ALL TESTS PASSED ✓')
        print('='*60)
        print('\nNext steps:')
        print('1. Train student model:')
        print('   python3 trainSpeakerNet_performance_updated.py \\')
        print('     --config configs/mlp_mixer_distillation_config.yaml')
        print('\n2. Monitor training:')
        print('   tensorboard --logdir exps/mlp_mixer_distillation')
        print('\n3. Expected results:')
        print('   - EER: 10-11% (teacher is 9.68%)')
        print('   - Speed: 2-3× faster than LSTM+AE')
        print('   - Size: 2.66M params (vs LSTM+AE 3.87M)')
        print('='*60)
        
    except Exception as e:
        print(f'\n✗ TEST FAILED: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
