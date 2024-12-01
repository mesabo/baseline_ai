#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14

# Activate conda environment
source ~/.bashrc
source activate fl_torch

# Check GPU allocation
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "No GPU devices available. Ensure the GPU node is correctly allocated."
else
    echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
fi

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit

# Default output directory
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# Run main.py with CLI arguments
python3 cnn_classification5.py \
    --model_type CNNConv2 \
    --optimizer adam \
    --batch_size 16 \
    --epochs 100 \
    --gpu 0 > "$OUTPUT_DIR/cnn_conv2_training.log" 2>&1

echo "Execution complete. Logs saved to $OUTPUT_DIR/cnn_conv2_training.log."
