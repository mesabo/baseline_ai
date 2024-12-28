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

# Configuration parameters
OPTIM="adam"                         # Optimizer (e.g., "adam", "sgd")
BS="16"                              # Batch size
EPOCH="90"                          # Number of epochs
MODEL_NAME="LSTM2CNN"                   # Model type (e.g., "LSTM", "GRU", "CNNConv1", etc.)
GPU="0"                              # GPU device ID (e.g., "0", "1")
LOOKBACK_DAYS="6"                   # Number of lookback days (input window size)
FORECAST_DAYS="1"                   # Number of forecast days (output size)
TARGET_COLUMN="Yield (kg)"           # Target column name in the dataset
RUNNING_MODE="train_val"             # Execution mode: train/validate and test (train_val) or test only (test)
DATASET_PATH="./data/synthetic_fishing_data.csv"  # Dataset file path
OUTPUT_DIR="./output/$MODEL_NAME"  # Output default dir

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Inform the user of the current settings
echo "Output Directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Optimizer: $OPTIM"
echo "Batch Size: $BS"
echo "Epochs: $EPOCH"
echo "GPU: $GPU"
echo "Lookback Days: $LOOKBACK_DAYS"
echo "Forecast Days: $FORECAST_DAYS"
echo "Target Column: $TARGET_COLUMN"
echo "Dataset Path: $DATASET_PATH"

# Run main.py with the specified arguments
python3 main.py \
    --model_type "$MODEL_NAME" \
    --optimizer "$OPTIM" \
    --batch_size "$BS" \
    --epochs "$EPOCH" \
    --gpu "$GPU" \
    --lookback_days "$LOOKBACK_DAYS" \
    --forecast_days "$FORECAST_DAYS" \
    --target_column "$TARGET_COLUMN" \
    --mode "$RUNNING_MODE"\
    --output_dir "$OUTPUT_DIR"\
    --dataset_path "$DATASET_PATH" > "$OUTPUT_DIR/$MODEL_NAME-$OPTIM-$BS-$EPOCH-$RUNNING_MODE.log" 2>&1

# Notify the user of completion
echo "Execution complete. Logs saved to $OUTPUT_DIR/$MODEL_NAME-$OPTIM-$BS-$EPOCH-$RUNNING_MODE.log."
