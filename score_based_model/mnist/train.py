import os
import numpy as np
import functools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

from score_based_model.kernel.model import ScoreNet
from score_based_model.kernel.functions import marginal_prob_std, diffusion_coeff


def loss_fn(model, x, marginal_prob_std, eps=1e-8):
    random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))
    return loss


def train(dataset, path_to_save, device="cuda"):
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    n_epochs = 1000
    batch_size = 64
    lr = 1e-4

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    optimizer = Adam(score_model.parameters(), lr=lr)

    avg_loss = 0.0
    num_items = 0
    tqdm_epoch = tqdm(range(n_epochs))
    for epoch in tqdm_epoch:
        for x, y, z in data_loader:
            x = x.to(device)
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        if np.isnan(avg_loss / num_items):
            break
        # Print the averaged training loss so far.
        tqdm_epoch.set_description("Average Loss: {:5f}".format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), os.path.join(path_to_save, "checkpoint.pth"))
