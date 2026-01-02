#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Test Validation Phase - Verify validation/testing works correctly
This script tests the validation phase independently without training
"""

import sys, time, os, argparse
import yaml
import numpy
import torch
from tuneThreshold import *
from SpeakerNet_performance_updated import *
from DatasetLoader_performance_updated import *

print("="*80)
print("VALIDATION PHASE TEST")
print("="*80)

# Parse arguments
parser = argparse.ArgumentParser(description = "Test Validation Phase")
parser.add_argument('--config',         type=str,   default='configs/experiment_01_performance_updated.yaml',   help='Config YAML file')
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights (optional)')
parser.add_argument('--test_list',      type=str,   default="data/list_test_all_formated_cleaned.txt",   help='Evaluation list')
parser.add_argument('--test_path',      type=str,   default="/mnt/ricproject3/2025/data/rearranged_voxceleb1", help='Absolute path to the test set')
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

args = parser.parse_args()

# Load config if provided
if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            args.__dict__[k] = v

print(f"\nConfig: {args.config}")
print(f"Test list: {args.test_list}")
print(f"Test path: {args.test_path}")
print(f"Model: {args.model}")
print(f"Encoder: {args.encoder_type}")
print(f"Embedding size: {args.nOut}")
print(f"Initial model: {args.initial_model if args.initial_model else 'Random initialization (testing format only)'}")

# Check if test files exist
if not os.path.exists(args.test_list):
    print(f"\n❌ ERROR: Test list not found: {args.test_list}")
    sys.exit(1)

if not os.path.exists(args.test_path):
    print(f"\n❌ ERROR: Test path not found: {args.test_path}")
    sys.exit(1)

print(f"\n✓ Test list exists: {args.test_list}")
print(f"✓ Test path exists: {args.test_path}")

# Initialize model
print("="*80)
print("INITIALIZING MODEL")
print("="*80)

# Initialize the SpeakerNet model
s = SpeakerNet(**vars(args))

# Move model to GPU if available
if torch.cuda.is_available():
    args.gpu = 0
    torch.cuda.set_device(args.gpu)
    s = WrappedModel(s).cuda(args.gpu)
    print(f"✓ Model moved to GPU: cuda:{args.gpu}")
else:
    args.gpu = None
    print("⚠ No GPU available, using CPU")

# Load initial model if provided
if args.initial_model and os.path.exists(args.initial_model):
    print(f"Loading model weights from: {args.initial_model}")
    s.__S__.loadParameters(args.initial_model)
    print("✓ Model weights loaded successfully")
else:
    print("⚠ No initial model provided - using random initialization")
    print("⚠ This will test the validation format, but metrics won't be meaningful")

# Create trainer with the wrapped model
trainer = ModelTrainer(s, **vars(args))

# Load model if provided (optional for format testing)
if args.initial_model != "":
    if os.path.exists(args.initial_model):
        print(f"Loading model from: {args.initial_model}")
        trainer.loadParameters(args.initial_model)
        print("✓ Model loaded successfully")
    else:
        print(f"⚠ Model file not found: {args.initial_model}")
        print("⚠ Continuing with random initialization (testing format only)")
else:
    print("⚠ No initial model provided - using random initialization")
    print("⚠ This will test the validation format, but metrics won't be meaningful")

# Test validation phase
print("\n" + "="*80)
print("RUNNING VALIDATION PHASE")
print("="*80)

try:
    print("\nStep 1: Loading test data and computing embeddings...")
    sc, lab, trials = trainer.evaluateFromList(**vars(args))
    print(f"✓ Embeddings computed successfully")
    print(f"  - Scores shape: {sc.shape if hasattr(sc, 'shape') else len(sc)}")
    print(f"  - Labels shape: {lab.shape if hasattr(lab, 'shape') else len(lab)}")
    
    print("\nStep 2: Computing EER and threshold...")
    result = tuneThresholdfromScore(sc, lab, [1, 0.1])
    current_eer = float(result[1])  # Convert to float
    current_threshold = result[2]
    
    # Handle threshold which might be an array or tuple
    if hasattr(current_threshold, '__iter__') and not isinstance(current_threshold, str):
        threshold_val = float(current_threshold[0]) if len(current_threshold) > 0 else 0.0
    else:
        threshold_val = float(current_threshold)
    
    print(f"✓ EER computed: {current_eer:2.4f}%")
    print(f"  - Threshold: {threshold_val:f}")
    print(f"  - Type of EER: {type(current_eer)}")
    print(f"  - Type of threshold: {type(threshold_val)}")
    
    print("\nStep 3: Computing MinDCF...")
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
    mindcf = float(mindcf)  # Convert to float
    
    print(f"✓ MinDCF computed: {mindcf:2.5f}")
    print(f"  - Type of MinDCF: {type(mindcf)}")
    
    print("\nStep 4: Testing formatted output...")
    formatted_output = f'{time.strftime("%Y-%m-%d %H:%M:%S")} Epoch TEST, VEER {current_eer:2.4f}, MinDCF {mindcf:2.5f}, Threshold {threshold_val:f}'
    print(f"✓ Formatted output successful:")
    print(f"  {formatted_output}")
    
    print("\n" + "="*80)
    print("✅ VALIDATION PHASE TEST SUCCESSFUL!")
    print("="*80)
    print("\nAll validation steps completed without errors:")
    print("  ✓ Test data loading")
    print("  ✓ Embedding computation")
    print("  ✓ EER calculation")
    print("  ✓ MinDCF calculation")
    print("  ✓ Metric formatting")
    print("  ✓ Output printing")
    
    if args.initial_model == "":
        print("\n⚠ Note: Random initialization was used, so metrics are not meaningful.")
        print("   Run again with --initial_model to test with a trained model.")
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ VALIDATION PHASE TEST FAILED!")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("TEST COMPLETE - Training can proceed safely")
print("="*80)
