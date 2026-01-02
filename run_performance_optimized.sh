#!/bin/bash

# Quick Start Script for Performance Optimized Trainer
# This script helps you easily run the optimized version

echo "=========================================="
echo "VoxCeleb Trainer - Performance Optimized"
echo "=========================================="
echo ""

# Check if config argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_performance_optimized.sh <mode> [options]"
    echo ""
    echo "Modes:"
    echo "  train          - Run optimized training"
    echo "  benchmark      - Benchmark performance"
    echo "  compare        - Compare original vs optimized"
    echo "  test           - Test dataloader"
    echo ""
    echo "Examples:"
    echo "  ./run_performance_optimized.sh train"
    echo "  ./run_performance_optimized.sh benchmark"
    echo "  ./run_performance_optimized.sh compare"
    echo "  ./run_performance_optimized.sh test"
    exit 1
fi

MODE=$1

case $MODE in
    train)
        echo "🚀 Running OPTIMIZED training..."
        echo "Config: configs/experiment_01_performance_updated.yaml"
        echo ""
        python trainSpeakerNet_performance_updated.py \
            --config configs/experiment_01_performance_updated.yaml \
            --mixedprec
        ;;
    
    benchmark)
        echo "📊 Benchmarking OPTIMIZED version..."
        echo ""
        python benchmark_performance.py \
            --config configs/experiment_01_performance_updated.yaml \
            --num_batches 100
        ;;
    
    compare)
        echo "📊 Comparing ORIGINAL vs OPTIMIZED..."
        echo ""
        python benchmark_performance.py \
            --config configs/experiment_01_performance_updated.yaml \
            --num_batches 100 \
            --compare
        ;;
    
    test)
        echo "🧪 Testing dataloader..."
        echo ""
        python test_dataloader.py
        ;;
    
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Valid modes: train, benchmark, compare, test"
        exit 1
        ;;
esac

echo ""
echo "✓ Complete!"
