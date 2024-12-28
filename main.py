from models.cnn_classifications import MyClassification
from models.cnns import cnn_image as cnn_img
from utils.argument_parser import parse_arguments
from models.rnn_regressions import RunRNNModel

if __name__ == "__main__":
    print("Code execution.....")

    # Parse arguments using the external argument parser
    args = parse_arguments()

    # Run CNN Image models
    if args.model_type == "RunCNNImageModel1":
        cnn_img.RunCNNImageModel1()
    elif args.model_type == "RunCNNImageModel2":
        cnn_img.RunCNNImageModel2()

    # Run regression or CNN classification models
    elif args.model_type in ["SimpleRegression", "CNNConv1", "CNNConv2"]:
        MyClassification(args)

    # Run RNN-based models (LSTM, GRU, BiLSTM, BiGRU)
    elif args.model_type in ["LSTM", "GRU", "BiLSTM", "BiGRU", "LSTM2", "LSTM2CNN"]:
        rnn_runner = RunRNNModel(args)
        rnn_runner.run()

    # Handle invalid model types
    else:
        print(f"Model type {args.model_type} is not implemented!")