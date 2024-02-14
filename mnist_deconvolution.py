import os
from multiprocessing import Process
import warnings
import functools
import torch
from torchvision.io import read_image
from natsort import natsorted

from app.models.functions import normalize
from app.utils import create_results_dir, save_originals, run
from app.config import Config
from app.deconvolution import optimize
from score_based_model.mnist.utils import get_mnist_score_model
from score_based_model.mnist.functions import marginal_prob_std


def mnist_deconvolution(params, path_to_save, blur_image_path, sharp_image_path, blur_kernel_path, noise_std, device):
    # load sharp image
    try:
        sharp_image = read_image(sharp_image_path)
        sharp_image = normalize(sharp_image)
    except FileNotFoundError as e:
        print(e)

    # load kernel image (gray scale)
    try:
        kernel_image = read_image(blur_kernel_path)
        # remove noise from kernel
        kernel_image = torch.where(kernel_image < 10, 0, kernel_image).to(device)
        kernel_image = normalize(kernel_image)
        kernel_image.squeeze_()
    except FileNotFoundError as e:
        print(e)

    # load blur image
    try:
        blur_image = read_image(blur_image_path)
        blur_image = normalize(blur_image)
        # add noise
        blur_image = blur_image + torch.normal(0, noise_std, size=blur_image.size())
        blur_image = torch.clip(blur_image, 0, 1)
    except FileNotFoundError as e:
        print(e)

    # save original image and kernel.
    fname = blur_image_path.split("/")[-1]
    save_originals(fname, path_to_save, sharp_image.detach().clone(), kernel_image.detach().clone(), blur_image.detach().clone())

    # load score model
    image_ckpt_path = "score_based_model/checkpoints/mnist/checkpoint.pth"
    ## image
    try:
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=25.0, device=device)
        image_score_model = get_mnist_score_model(image_ckpt_path, marginal_prob_std_fn, device=device)
        # image_score_model.share_memory()
    except FileNotFoundError as e:
        print(e)
        print("Specify the checkpoint path.")

    image_size = (1, 28, 28)
    print(f"Deconvolution(use known kernel) {fname} ({int(noise_std * 100)} perc noise).")
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

    # load config.
    params = Config.get_conf()
    noise_std = params["noise_std"]
    device = params["device"]

    # create results dir.
    folder_to_save = create_results_dir(
        class_name=f"deconv_mnist ({int(noise_std * 100)}_perc_noise)",
        results_path="deconv_results/mnist",
    )

    # get image paths.
    blur_images_folder = "dataset/blured_mnist/blur_images"
    blur_kernels_folder = "dataset/blured_mnist/blur_kernels"
    sharp_images_folder = "dataset/blured_mnist/sharp_images"
    blur_image_paths = natsorted(os.listdir(blur_images_folder))
    blur_kernel_paths = natsorted(os.listdir(blur_kernels_folder))
    sharp_image_paths = natsorted(os.listdir(sharp_images_folder))

    id = params["id"]
    N = params["epoch_num"]
    assert id + N <= len(blur_image_paths), "The sum of id and epoch_num should be set to less than the total number of data."
    for b_path, k_path, i_path in zip(blur_image_paths[id:], sharp_image_paths[id:], blur_kernel_paths[id:]):
        blur_image_path = os.path.join(blur_images_folder, b_path)
        sharp_image_path = os.path.join(sharp_images_folder, i_path)
        blur_kernel_path = os.path.join(blur_kernels_folder, k_path)

        mnist_deconvolution(
            params,
            folder_to_save,
            blur_image_path,
            sharp_image_path,
            blur_kernel_path,
            noise_std,
            device,
        )
        N -= 1
        if N == 0:
            break
    #     # make process.
    #     process = Process(
    #         target=mnist_deconvolution,
    #         args=(
    #             params,
    #             folder_to_save,
    #             blur_image_path,
    #             sharp_image_path,
    #             blur_kernel_path,
    #             noise_std,
    #             device,
    #         ),
    #     )
    #     processes.append(process)
    #     # run processes.
    #     if len(processes) == processe_num:
    #         run(processes)
    #         processes = []
    # run(processes)
