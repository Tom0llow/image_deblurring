import sys

from score_based_model.mnist.utils import create_dataset
from score_based_model.mnist.train import train
from score_based_model.mnist.sampling import sampling


if __name__ == "__main__":
    args = sys.argv

    if len(args) < 2:
        raise Exception("Arguments are too short.")

    elif args[1] == "--train":
        print("Run train & sampling")
        # train
        folder = "./dataset"
        folder_to_save = "./checkpoints/mnist"
        dataset = create_dataset(folder)
        train(dataset, folder_to_save)
        # sampling
        ckpt_path = "./score_based_model/checkpoints/mnist/checkpoint.pth"
        path_to_save = "./image/sample"
        filename = "mnist.png"
        sampling(ckpt_path, path_to_save, filename, device="cuda")

    elif args[1] == "--sampling":
        # sampling
        print("Run only sampling")
        ckpt_path = "./score_based_model/checkpoints/mnist/checkpoint.pth"
        path_to_save = "./image/sample"
        filename = "mnist.png"
        sampling(ckpt_path, path_to_save, filename, device="cuda")

    else:
        raise Exception("Please input arguments '--train' or '--sampling'.")
