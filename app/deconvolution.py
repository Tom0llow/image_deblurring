import torch
from torchvision.io import read_image

from score_based_model.celebahq_256.utils import get_image_score_model
from app.non_blind_optimizer import optimize
from app.utils import save_estimateds


# Deconvolution(use known kernel)
def run(params, path_to_save, blur_image_path, kernel_image_path, image_ckpt_path, device="cuda"):
    # load blur image
    try:
        blur_image = read_image(blur_image_path)
    except FileNotFoundError as e:
        print(e)

    # load kernel image (gray scale)
    try:
        kernel_image = read_image(kernel_image_path)
    except FileNotFoundError as e:
        print(e)

    # load score model
    # checkpoints
    ## image
    try:
        sde, image_score_model = get_image_score_model(image_ckpt_path, device=device)
    except FileNotFoundError as e:
        print(e)
        print("Specify the checkpoint path.")

    estimated_i = optimize(
        blur_image,
        kernel_image,
        image_score_model,
        lambda_=params["lambda_"],
        eta_=params["eta_"],
        fname=blur_image_path.split("/")[-1],
        path_to_save=path_to_save,
        save_interval=params["save_interval"],
        num_steps=params["num_steps"],
        num_scales=params["num_scales"],
        batch_size=params["batch_size"],
        device=device,
    )

    # save
    save_estimateds(
        fname=blur_image_path.split("/")[-1],
        path_to_save=path_to_save,
        estimated_i=estimated_i,
    )
