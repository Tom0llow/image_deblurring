import torch
from torchvision.io import read_image
import numpy as np
import functools

from score_based_model.kernel.functions import marginal_prob_std
from score_based_model.kernel.utils import get_kernel_score_model
from score_based_model.celebahq_256.utils import get_image_score_model
from app.model import optimize
from app.config import Config
from app.utils import save_estimateds


def run(path_to_save, blur_image_path, image_ckpt_path, kernel_ckpt_path, device="cuda"):
    # load blur image
    try:
        blur_image = read_image(blur_image_path)
    except FileNotFoundError as e:
        print(e)

    # load score model
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    # checkpoints
    ## image
    try:
        sde, image_score_model = get_image_score_model(image_ckpt_path, device=device)
    except FileNotFoundError as e:
        print(e)
        print("Specify the checkpoint path.")
    ## kernel
    try:
        kernel_score_model = get_kernel_score_model(kernel_ckpt_path, marginal_prob_std_fn, device=device)
    except FileNotFoundError as e:
        print(e)
        print("Specify the checkpoint path.")

    # load config
    params = Config.get_conf()

    estimated_i, estimated_k = optimize(
        blur_image,
        image_score_model,
        kernel_score_model,
        marginal_prob_std_fn,
        alpha_=params["alpha_"],
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
        estimated_k=estimated_k,
    )
