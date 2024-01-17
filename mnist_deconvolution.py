import torch
from torchvision.io import read_image
import functools

from app.utils import create_results_dir, save_originals
from app.config import Config
from app.deconvolution import optimize
from score_based_model.mnist.utils import get_mnist_score_model
from score_based_model.mnist.functions import marginal_prob_std


if __name__ == "__main__":
    create_results_dir(class_name="mnist_deconvolution")
    folder_to_save = "image/results/mnist_deconvolution"

    blur_image_path = "dataset/blured_mnist/blur_images/img_473.jpg"
    blur_kernel_path = "dataset/blured_mnist/blur_kernels/img_473.jpg"
    sharp_image_path = "dataset/blured_mnist/sharp_images/img_473.jpg"
    # save original images(kernels)
    fname = blur_image_path.split("/")[-1]
    save_originals(fname, paths=[blur_image_path, sharp_image_path, blur_kernel_path], path_to_save=folder_to_save)

    image_ckpt_path = "score_based_model/checkpoints/mnist/checkpoint.pth"
    kernel_ckpt_path = "score_based_model/checkpoints/blur_kernel/checkpoint.pth"

    # load config
    params = Config.get_conf()
    device = params["device"]

    # load blur image
    try:
        blur_image = read_image(blur_image_path)
        fname = blur_image_path.split("/")[-1]
    except FileNotFoundError as e:
        print(e)

    # load kernel image (gray scale)
    try:
        kernel_image = read_image(blur_kernel_path)
    except FileNotFoundError as e:
        print(e)
    # load score model
    ## image
    try:
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25.0, device=device)
        image_score_model = get_mnist_score_model(image_ckpt_path, marginal_prob_std_fn, device=device)
    except FileNotFoundError as e:
        print(e)
        print("Specify the checkpoint path.")

    image_size = (1, 28, 28)
    print("Deconvolution(use known kernel)")
    optimize(
        blur_image,
        kernel_image,
        image_size,
        image_score_model,
        lambda_=params["lambda_"],
        eta_=params["eta_"],
        fname=fname,
        path_to_save=folder_to_save,
        save_interval=params["save_interval"],
        num_steps=params["num_steps"],
        num_scales=params["num_scales"],
        batch_size=params["batch_size"],
        device=device,
    )
