import os
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

from score_based_model.kernel.model import ScoreNet
from score_based_model.kernel.dataset import Kernel_Img_Dataset, ImageTransform


def get_kernel_score_model(ckpt_path, marginal_prob_std_fn, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    score_model.load_state_dict(ckpt)

    print("Loaded kernel score model.")
    return score_model


def save_samples(samples, sample_batch_size, path_to_save, filename):
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)), pad_value=3)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.savefig(os.path.join(path_to_save, filename))


def create_dataset(path):
    folder = "./dataset/RandomMotionBlur"
    train_img_list = []
    for path in os.listdir(folder + "/train"):
        train_img_list.append(os.path.join(folder + "/train", path))

    org_dataset = Kernel_Img_Dataset(file_list=train_img_list, transform=ImageTransform())
    return org_dataset
