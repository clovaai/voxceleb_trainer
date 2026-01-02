#!/bin/bash

##
## Training Script: MLP-Mixer with Raw Waveform Input + Knowledge Distillation
##
## Experiment: P3 - Raw Waveform + Distillation
## Purpose: Test if raw waveform + teacher knowledge matches mel-based V2 (10.32% EER)
## Expected: 10.5-11.5% EER (comparable to mel-based distillation)
##

echo "========================================================================"
echo "Training MLP-Mixer with Raw Waveform Input + Distillation"
echo "========================================================================"
echo "Experiment: P3 - Raw Waveform + Knowledge Distillation"
echo "Student: MLP-Mixer + SincNet (3.48M parameters, raw waveform)"
echo "Teacher: LSTM+Autoencoder (3.87M parameters, 9.68% EER)"
echo "Distillation: α=0.7, T=4.0, Cosine loss"
echo "Dataset: Mini VoxCeleb2 (30K samples)"
echo "Expected EER: 10.5-11.5% (vs mel-based V2: 10.32%)"
echo "========================================================================"
echo ""

# Activate environment
source ~/anaconda3/bin/activate 2025_colvaai

# Check teacher checkpoint exists
TEACHER_CKPT="exps/lstm_autoencoder_distillation/model/best_state.model"
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: Teacher checkpoint not found at $TEACHER_CKPT"
    echo "Please ensure LSTM+Autoencoder teacher model is trained first."
    exit 1
fi

echo "✓ Teacher checkpoint found: $TEACHER_CKPT"
echo ""

# Training command
python3 trainSpeakerNet_distillation.py \
    --config configs/mlp_mixer_rawwaveform_distillation.yaml \
    --mixedprec

echo ""
echo "========================================================================"
echo "Training completed!"
echo "Results saved to: exps/mlp_mixer_rawwaveform_distillation/"
echo "========================================================================"
echo ""
echo "Analysis steps:"
echo "1. Check best EER in exps/mlp_mixer_rawwaveform_distillation/result/scores.txt"
echo "2. Compare with mel-based V2 (10.32% EER):"
echo "   - If EER ≤ 10.5%: Raw waveform is comparable/better ✓"
echo "   - If EER > 11.5%: Mel preprocessing remains superior"
echo "3. Document results in research log"
echo "========================================================================"
