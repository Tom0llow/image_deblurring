import os
import torch
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt

from score_based_model.mnist.model import ScoreNet
from score_based_model.mnist.dataset import Mnist_Img_Dataset, ImageTransform


def get_mnist_score_model(ckpt_path, marginal_prob_std_fn, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    score_model.load_state_dict(ckpt)

    print("Loaded mnist score model.")
    return score_model


def save_samples(samples, sample_batch_size, path_to_save, filename):
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)), pad_value=3)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.savefig(os.path.join(path_to_save, filename))


def create_dataset(folder):
    train_img_list = []
    for path in os.listdir(folder + "/train"):
        train_img_list.append(os.path.join(folder + "/train", path))

    org_dataset = Mnist_Img_Dataset(file_list=train_img_list, transform=ImageTransform())
    return org_dataset
