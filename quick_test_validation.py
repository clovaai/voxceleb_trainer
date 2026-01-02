#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Quick Validation Test - Fast test of validation phase with GPU verification
Tests only a small subset of data to verify the pipeline works
"""

import sys, time, os, argparse
import yaml
import numpy
import torch
from tuneThreshold import *
from SpeakerNet_performance_updated import *
from DatasetLoader_performance_updated import *

print("="*80)
print("QUICK VALIDATION TEST - GPU & Format Verification")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser(description="Quick Validation Test")
parser.add_argument('--config',         type=str,   default='configs/experiment_01_performance_updated.yaml',   help='Config YAML file')
parser.add_argument('--initial_model',  type=str,   required=True,  help='Initial model weights')
parser.add_argument('--test_list',      type=str,   default="data/list_test_all_formated_cleaned.txt",   help='Evaluation list')
parser.add_argument('--test_path',      type=str,   default="/mnt/ricproject3/2025/data/rearranged_voxceleb1", help='Absolute path to the test set')
parser.add_argument('--max_pairs',      type=int,   default=1000,   help='Maximum test pairs to use (for quick test)')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing')
parser.add_argument('--model',          type=str,   default="ResNetSE34V2",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--trainfunc',      type=str,   default="aamsoftmax",     help='Loss function')
parser.add_argument('--nClasses',       type=int,   default=5991,   help='Number of speakers')
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale')
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='DCF p_target')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='DCF c_miss')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='DCF c_fa')
parser.add_argument('--optimizer',      type=str,   default="adam", help='Optimizer')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument('--lr_decay',       type=float, default=0.95,   help='Learning rate decay')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay')
parser.add_argument('--gpu',            type=int,   default=0,      help='GPU index')
parser.add_argument('--mixedprec',      dest='mixedprec', action='store_true', default=True, help='Enable mixed precision')
parser.add_argument('--test_interval',  type=int,   default=3,      help='Test interval')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum epochs')
parser.add_argument('--nDataLoaderThread', type=int, default=8,     help='Number of loader threads')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--eval_batch_size', type=int,  default=128,    help='Evaluation batch size for faster processing')

args = parser.parse_args()

# Save initial_model before config override
initial_model_arg = args.initial_model

# Load config if provided
if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__ and k != 'initial_model':  # Don't override initial_model from config
            args.__dict__[k] = v

# Restore initial_model argument
args.initial_model = initial_model_arg

print(f"\nConfig: {args.config}")
print(f"Model: {args.initial_model}")
print(f"Test list: {args.test_list}")
print(f"Test path: {args.test_path}")
print(f"Max pairs for quick test: {args.max_pairs}")

# Check if files exist
if not os.path.exists(args.initial_model):
    print(f"\n❌ ERROR: Model not found: {args.initial_model}")
    sys.exit(1)

if not os.path.exists(args.test_list):
    print(f"\n❌ ERROR: Test list not found: {args.test_list}")
    sys.exit(1)

if not os.path.exists(args.test_path):
    print(f"\n❌ ERROR: Test path not found: {args.test_path}")
    sys.exit(1)

# Check GPU
if not torch.cuda.is_available():
    print("\n❌ ERROR: CUDA not available!")
    sys.exit(1)

print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA Version: {torch.version.cuda}")
print(f"✓ PyTorch Version: {torch.__version__}")

# Create a small test list
print(f"\n📝 Creating subset of test list ({args.max_pairs} pairs)...")
with open(args.test_list, 'r') as f:
    all_lines = f.readlines()

total_pairs = len(all_lines)
print(f"Total pairs in full list: {total_pairs}")

# Create temporary test list
temp_test_list = '/tmp/quick_test_list.txt'
with open(temp_test_list, 'w') as f:
    f.writelines(all_lines[:args.max_pairs])

args.test_list = temp_test_list
print(f"✓ Created temporary test list: {temp_test_list}")

# Initialize model
print("\n" + "="*80)
print("INITIALIZING MODEL & LOADING WEIGHTS")
print("="*80)

start_time = time.time()

# Initialize the SpeakerNet model
s = SpeakerNet(**vars(args))

# Move model to GPU
args.gpu = 0
torch.cuda.set_device(args.gpu)
s = WrappedModel(s).cuda(args.gpu)
print(f"✓ Model moved to GPU: cuda:{args.gpu}")

init_time = time.time() - start_time
print(f"✓ Model initialization time: {init_time:.2f}s")

# Create trainer
trainer = ModelTrainer(s, **vars(args))

# Load model weights
print(f"Loading weights from: {args.initial_model}")
load_start = time.time()
trainer.loadParameters(args.initial_model)
load_time = time.time() - load_start
print(f"✓ Model weights loaded in {load_time:.2f}s")

# Run validation
print("\n" + "="*80)
print("RUNNING VALIDATION ON GPU")
print("="*80)

try:
    print(f"\n🚀 Computing embeddings for {args.max_pairs} pairs...")
    val_start = time.time()
    
    sc, lab, trials = trainer.evaluateFromList(**vars(args))
    
    val_time = time.time() - val_start
    print(f"✓ Embeddings computed in {val_time:.2f}s")
    print(f"  Speed: {args.max_pairs/val_time:.1f} pairs/second")
    print(f"  Scores shape: {sc.shape if hasattr(sc, 'shape') else len(sc)}")
    print(f"  Labels shape: {lab.shape if hasattr(lab, 'shape') else len(lab)}")
    
    print("\n📊 Computing metrics...")
    metric_start = time.time()
    
    # Compute EER
    result = tuneThresholdfromScore(sc, lab, [1, 0.1])
    current_eer = float(result[1])
    current_threshold = result[2]
    
    if hasattr(current_threshold, '__iter__') and not isinstance(current_threshold, str):
        threshold_val = float(current_threshold[0]) if len(current_threshold) > 0 else 0.0
    else:
        threshold_val = float(current_threshold)
    
    # Compute MinDCF
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
    mindcf = float(mindcf)
    
    metric_time = time.time() - metric_start
    print(f"✓ Metrics computed in {metric_time:.2f}s")
    
    # Format output
    formatted_output = f'{time.strftime("%Y-%m-%d %H:%M:%S")} Epoch TEST, VEER {current_eer:2.4f}, MinDCF {mindcf:2.5f}, Threshold {threshold_val:f}'
    
    print("\n" + "="*80)
    print("✅ VALIDATION TEST SUCCESSFUL!")
    print("="*80)
    print(f"\n{formatted_output}")
    print(f"\nPerformance:")
    print(f"  Initialization: {init_time:.2f}s")
    print(f"  Embedding computation: {val_time:.2f}s ({args.max_pairs/val_time:.1f} pairs/s)")
    print(f"  Metric computation: {metric_time:.2f}s")
    print(f"  Total: {init_time + val_time + metric_time:.2f}s")
    
    print(f"\n⚠️  Note: Tested on {args.max_pairs} pairs (subset)")
    print(f"   Full test set has {total_pairs} pairs")
    print(f"   Estimated full test time: {(val_time + metric_time) * total_pairs / args.max_pairs:.1f}s")
    
    # GPU memory usage
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ VALIDATION TEST FAILED!")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Clean up temp file
    if os.path.exists(temp_test_list):
        os.remove(temp_test_list)

print("\n" + "="*80)
print("🎉 TEST COMPLETE - GPU & Pipeline Working!")
print("="*80)
