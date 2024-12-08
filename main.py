from models.cnn_classifications import MyClassification
from models.cnns import cnn_image as cnn_img
from utils.argument_parser import parse_arguments

if __name__=="__main__":
    print("Code execution.....")

    # Parse arguments using the external argument parser
    args = parse_arguments()

    # run CNN Image model
    if args.model_type == "RunCNNImageModel1":
        cnn_img.RunCNNImageModel1()
    elif args.model_type == "RunCNNImageModel2":
        cnn_img.RunCNNImageModel2()
    elif args.model_type in ["SimpleRegression", "CNNConv1", "CNNConv2"]:
        MyClassification(args)
    else:
        print("Model type not found!")
