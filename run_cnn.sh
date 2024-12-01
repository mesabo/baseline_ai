#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14

# Activate conda environment
source ~/.bashrc
source activate time_series

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
MODEL_NAME="CNNConv2"
mkdir -p "$OUTPUT_DIR"

# Run main.py with CLI arguments
python3 main.py \
    --model_type MODEL_NAME \
    --optimizer adam \
    --batch_size 32 \
    --epochs 100 \
    --gpu 0 > "$OUTPUT_DIR/cnn-$MODEL_NAME-training.log" 2>&1

echo "Execution complete. Logs saved to $OUTPUT_DIR/cnn-$MODEL_NAME-training.log."
