import os
from multiprocessing import Process
import torch
from torchvision.io import read_image
import functools
import warnings

from app.utils import create_results_dir, save_originals, run
from app.config import Config
from app.deconvolution import optimize
from score_based_model.mnist.utils import get_mnist_score_model
from score_based_model.mnist.functions import marginal_prob_std


def mnist_deconvolution(params, path_to_save, blur_image_path, sharp_image_path, blur_kernel_path, device):
    # save original image and kernel.
    fname = blur_image_path.split("/")[-1]
    save_originals(fname, paths=[blur_image_path, sharp_image_path, blur_kernel_path], path_to_save=path_to_save)

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
    image_ckpt_path = "score_based_model/checkpoints/mnist/checkpoint.pth"
    ## image
    try:
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25.0, device=device)
        image_score_model = get_mnist_score_model(image_ckpt_path, marginal_prob_std_fn, device=device)
    except FileNotFoundError as e:
        print(e)
        print("Specify the checkpoint path.")

    image_size = (1, 28, 28)
    print(f"Deconvolution(use known kernel) {fname}")
    optimize(
        blur_image,
        kernel_image,
        image_size,
        image_score_model,
        lambda_=params["lambda_"],
        eta_=params["eta_"],
        fname=fname,
        path_to_save=path_to_save,
        save_interval=params["save_interval"],
        num_steps=params["num_steps"],
        num_scales=params["num_scales"],
        batch_size=params["score_batch_size"],
        patience=params["patience"],
        device=device,
    )


if __name__ == "__main__":
    # disable warnign message.
    warnings.simplefilter("ignore")

    # create results dir.
    folder_to_save = create_results_dir(class_name="mnist_deconvolution")

    # get image paths.
    blur_images_folder = "dataset/blured_mnist/blur_images"
    blur_kernels_folder = "dataset/blured_mnist/blur_kernels"
    sharp_images_folder = "dataset/blured_mnist/sharp_images"
    blur_image_paths = os.listdir(blur_images_folder)
    blur_kernel_paths = os.listdir(blur_kernels_folder)
    sharp_image_paths = os.listdir(sharp_images_folder)

    # load config
    params = Config.get_conf()
    device = params["device"]

    processes = []
    processe_num = params["process_num"]
    N = params["data_num"]
    assert N <= len(blur_image_paths) - 1, "data_num should be set to less than the total number of data."
    for b_path, k_path, i_path in zip(blur_image_paths[:N], sharp_image_paths[:N], blur_kernel_paths[:N]):
        blur_image_path = os.path.join(blur_images_folder, b_path)
        sharp_image_path = os.path.join(sharp_images_folder, i_path)
        blur_kernel_path = os.path.join(blur_kernels_folder, k_path)

        # make process.
        process = Process(
            target=mnist_deconvolution,
            args=(
                params,
                folder_to_save,
                blur_image_path,
                sharp_image_path,
                blur_kernel_path,
                device,
            ),
        )
        processes.append(process)
        # run processes.
        if len(processes) == processe_num:
            run(processes)
            processes = []
    run(processes)
