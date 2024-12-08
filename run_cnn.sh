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
OPTIM="adam"
BS="32"
EPOCH="100"
MODEL_NAME="CNNConv2" # ["RunCNNImageModel1", "RunCNNImageModel2", "SimpleRegression", "CNNConv1", "CNNConv2"]
mkdir -p "$OUTPUT_DIR"

# Run main.py with CLI arguments

echo "output dir: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"

python3 main.py \
    --model_type "$MODEL_NAME" \
    --optimizer "$OPTIM" \
    --batch_size "$BS" \
    --epochs "$EPOCH" \
    --gpu "0" > "$OUTPUT_DIR/$MODEL_NAME-$OPTIM-$BS-$EPOCH-training.log" 2>&1

echo "Execution complete. Logs saved to $OUTPUT_DIR/$MODEL_NAME-$OPTIM-$BS-$EPOCH-training.log."
