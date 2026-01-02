#!/bin/bash

##
## Training Script: MLP-Mixer with Raw Waveform Input (Baseline - No Distillation)
##
## Experiment: P3 - Raw Waveform Input Testing (Baseline)
## Purpose: Validate raw waveform processing before applying distillation
## Expected: 12-14% EER (baseline without teacher knowledge)
##

echo "========================================================================"
echo "Training MLP-Mixer with Raw Waveform Input (Baseline)"
echo "========================================================================"
echo "Experiment: P3 - Raw Waveform Testing"
echo "Model: MLP-Mixer + SincNet (3.48M parameters)"
echo "Input: Raw waveform (16kHz, learnable filters)"
echo "Dataset: Mini VoxCeleb2 (30K samples)"
echo "Expected EER: 12-14% (baseline, no distillation)"
echo "========================================================================"
echo ""

# Activate environment
source ~/anaconda3/bin/activate 2025_colvaai

# Training command
python3 trainSpeakerNet.py \
    --config configs/mlp_mixer_rawwaveform_baseline.yaml \
    --mixedprec

echo ""
echo "========================================================================"
echo "Training completed!"
echo "Results saved to: exps/mlp_mixer_rawwaveform_baseline/"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Check validation EER in exps/mlp_mixer_rawwaveform_baseline/result/scores.txt"
echo "2. If EER < 14%, proceed with distillation training:"
echo "   bash train_mlp_mixer_rawwaveform_distillation.sh"
echo "========================================================================"
