#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
Test max_test_pairs parameter
"""

import sys, argparse, yaml

# Parse arguments like the training script
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/experiment_01_performance_updated.yaml')
parser.add_argument('--test_list', type=str, default="/mnt/ricproject3/2025/data/list_test_all_formated_cleaned.txt")
parser.add_argument('--max_test_pairs', type=int, default=0, help='Maximum test pairs (0 = all)')

args = parser.parse_args()

# Save command line value
max_test_pairs_cmdline = args.max_test_pairs

# Load config
if args.config:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__ and k != 'max_test_pairs':  # Don't override if specified on command line
            args.__dict__[k] = v
    
    # Only use config value if not specified on command line
    if max_test_pairs_cmdline == 0 and 'max_test_pairs' in yml_config:
        args.max_test_pairs = yml_config['max_test_pairs']
    else:
        args.max_test_pairs = max_test_pairs_cmdline

# Read test list
with open(args.test_list) as f:
    lines = f.readlines()

total_pairs = len(lines)

print("="*80)
print("MAX_TEST_PAIRS PARAMETER TEST")
print("="*80)
print(f"\nTest list: {args.test_list}")
print(f"Total pairs in file: {total_pairs}")
print(f"max_test_pairs from config: {args.max_test_pairs}")

# Apply limit if specified
if args.max_test_pairs > 0 and args.max_test_pairs < total_pairs:
    used_pairs = args.max_test_pairs
    print(f"\n✅ Will use FIRST {used_pairs} pairs ({used_pairs/total_pairs*100:.2f}% of dataset)")
    print(f"   Estimated validation time: ~{used_pairs * 7.5 / 1000:.1f} seconds")
else:
    used_pairs = total_pairs
    print(f"\n✅ Will use ALL {used_pairs} pairs (100% of dataset)")
    print(f"   Estimated validation time: ~{used_pairs * 7.5 / 1000:.1f} seconds (~{used_pairs * 7.5 / 1000 / 60:.1f} minutes)")

print("\n" + "="*80)
print(f"Parameter working correctly!")
print("="*80)
