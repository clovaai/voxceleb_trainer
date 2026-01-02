#!/bin/bash

# Training script for MLP-Mixer with Knowledge Distillation
# Uses separate distillation training scripts (zero impact on existing models)

cd /mnt/ricproject3/2025/Colvaiai/voxceleb_trainer

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║       MLP-MIXER KNOWLEDGE DISTILLATION TRAINING                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Training Configuration:"
echo "  - Student: MLP-Mixer (2.66M params)"
echo "  - Teacher: LSTM+Autoencoder (9.68% EER, epoch 57)"
echo "  - Distillation: α=0.5, T=4.0"
echo "  - Expected EER: 10-11%"
echo ""
echo "Using SEPARATE training scripts:"
echo "  ✓ trainSpeakerNet_distillation.py (NEW)"
echo "  ✓ SpeakerNet_distillation.py (NEW)"
echo "  ✓ Existing models UNAFFECTED"
echo ""
echo "Press Ctrl+C to cancel, or wait 3 seconds to start..."
sleep 3

# Activate conda environment
source ~/.bashrc
conda activate 2025_colvaai

# Start training with distillation-aware script
python3 trainSpeakerNet_distillation.py \
  --config configs/mlp_mixer_distillation_config.yaml

echo ""
echo "Training complete! Check results:"
echo "  tail -f exps/mlp_mixer_distillation/result/scores.txt"
