import os

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from score_based_model.celebahq_256.utils import get_image_score_model


def sampling(sde, score_model, batch_size=64, snr=0.16, eps=1e-3, device="cuda"):
    x = sde.prior_sampling(shape=(batch_size, 3, 256, 256)).to(device)
    timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
    with torch.no_grad():
        for t in tqdm(timesteps):
            vec_t = torch.ones(batch_size, device=t.device) * t
            # Corrector step (Langevin MCMC)
            grad = score_model(x, vec_t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            x_mean = x + langevin_step_size * grad
            x = x_mean + torch.sqrt(2 * langevin_step_size) * noise

            # Predictor step (Euler-Maruyama)
            # dt = -1.0 / sde.N
            # z = torch.randn_like(x)
            # drift, diffusion = sde.sde(x, vec_t)
            # x_mean = x + drift * dt
            # x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        # The last step does not include any noise
        return x_mean


def run(image_ckpt_path, path_to_save, filename, device="cuda"):
    sde, image_score_model = get_image_score_model(image_ckpt_path, device=device)

    batch_size = 64

    samples = sampling(
        sde=sde,
        score_model=image_score_model,
        batch_size=batch_size,
        snr=0.16,
        eps=1e-5,
        device=device,
    )

    # save samples
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(batch_size)), pad_value=3)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.savefig(os.path.join(path_to_save, filename))
