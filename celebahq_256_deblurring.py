import torch
from torchvision.io import read_image
import functools

from app.utils import create_results_dir, save_originals
from app.config import Config


if __name__ == "__main__":
    create_results_dir(class_name="celebahq_256")
    folder_to_save = "image/results/celebahq_256"

    blur_image_path = "dataset/blured_celebahq_256/blur_images/000010.jpg"
    blur_kernel_path = "dataset/blured_celebahq_256/blur_kernels/000010.jpg"
    sharp_image_path = "dataset/blured_celebahq_256/sharp_images/000010.jpg"
    # save original images(kernels)
    fname = blur_image_path.split("/")[-1]
    save_originals(fname, paths=[blur_image_path, sharp_image_path, blur_kernel_path], path_to_save=folder_to_save)

    image_ckpt_path = "score_based_model/checkpoints/celebahq_256/checkpoint.pth"
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

    if params["blind"] is True:
        from app.blind_deconvolution import run
        from score_based_model.celebahq_256.utils import get_celebahq_256_score_model
        from score_based_model.kernel.functions import marginal_prob_std
        from score_based_model.kernel.utils import get_kernel_score_model

        # load score model
        ## image
        try:
            sde, image_score_model = get_celebahq_256_score_model(image_ckpt_path, device=device)
        except FileNotFoundError as e:
            print(e)
            print("Specify the checkpoint path.")
        ## kernel
        try:
            sigma = 25.0
            marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
            kernel_score_model = get_kernel_score_model(kernel_ckpt_path, marginal_prob_std_fn, device=device)
        except FileNotFoundError as e:
            print(e)
            print("Specify the checkpoint path.")

        print("Blind Deconvolution")
        run(
            params,
            fname,
            path_to_save=folder_to_save,
            blur_image=blur_image,
            image_score_model=image_score_model,
            kernel_score_model=kernel_score_model,
            image_size=(3, 256, 256),
            kernel_size=(1, 64, 64),
            device=device,
        )

    else:
        from app.deconvolution import run
        from score_based_model.celebahq_256.utils import get_celebahq_256_score_model

        # load kernel image (gray scale)
        try:
            kernel_image = read_image(blur_kernel_path)
        except FileNotFoundError as e:
            print(e)
        # load score model
        ## image
        try:
            sde, image_score_model = get_celebahq_256_score_model(image_ckpt_path, device=device)
        except FileNotFoundError as e:
            print(e)
            print("Specify the checkpoint path.")

        print("Deconvolution(use known kernel)")
        run(
            params,
            fname,
            path_to_save=folder_to_save,
            blur_image=blur_image,
            blur_kernel=kernel_image,
            image_score_model=image_score_model,
            image_size=(3, 256, 256),
            device=device,
        )
