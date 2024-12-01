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
        choices=["SimpleRegression", "CNNConv1", "CNNConv2"],
        default="SimpleRegression",
        help="Choose the model to run: SimpleRegression, CNNConv1, or CNNConv2",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        default="adam",
        help="Choose the optimizer: adam or sgd",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use (e.g., '0', '1', etc.)",
    )

    return parser.parse_args()
