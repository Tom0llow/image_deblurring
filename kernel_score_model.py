import sys

from score_based_model.kernel.utils import create_dataset
from score_based_model.kernel.train import train
from score_based_model.kernel.sampling import sampling


if __name__ == "__main__":
    args = sys.argv

    if len(args) < 2:
        raise Exception("Arguments are too short. Please input arguments '--train' or '--sampling'.")

    elif args[1] == "--train":
        print("Run train & sampling")
        # train
        folder = "./dataset/RandomMotionBlur"
        folder_to_save = "./score_based_model/checkpoints/blur_kernel"
        dataset = create_dataset(folder)
        train(dataset, folder_to_save)
        # sampling
        ckpt_path = "./score_based_model/checkpoints/blur_kernel/checkpoint.pth"
        path_to_save = "./image/sample"
        filename = "psf.png"
        sampling(ckpt_path, path_to_save, filename, device="cuda")

    elif args[1] == "--sampling":
        # sampling
        print("Run only sampling")
        ckpt_path = "./score_based_model/checkpoints/blur_kernel/checkpoint.pth"
        path_to_save = "./image/sample"
        filename = "psf.png"
        sampling(ckpt_path, path_to_save, filename, device="cuda")

    else:
        raise Exception("Please input arguments '--train' or '--sampling'.")
