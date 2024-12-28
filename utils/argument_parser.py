import argparse


def parse_arguments():
    """
    Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run Regression Models with Configurable Parameters")

    # Add arguments
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["RunCNNImageModel1", "RunCNNImageModel2", "SimpleRegression", "CNNConv1", "CNNConv2", "LSTM", "GRU",
"BiLSTM", "BiGRU", "LSTM2", "LSTM2CNN"],
        default="SimpleRegression",
        help="Choose the model to run: SimpleRegression, CNNConv1, or CNNConv2"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        default="adam",
        help="Choose the optimizer: adam or sgd"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use (e.g., '0', '1', etc.)"
    )
    parser.add_argument(
        "--lookback_days",
        type=int,
        default=30,
        help="Number of past days to use as input for the model"
    )
    parser.add_argument(
        "--forecast_days",
        type=int,
        default=10,
        help="Number of future days to predict"
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="Yield (kg)",
        help="Name of the target column in the dataset"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/synthetic_fishing_data.csv",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Path to the output files"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_val", "test"],
        default="train_val",
        help="Execution mode: train/validate and test (train_val) or test only (test)"
    )

    return parser.parse_args()