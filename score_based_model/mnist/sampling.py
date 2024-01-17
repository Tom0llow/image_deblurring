import numpy as np
import functools
from tqdm import tqdm
import torch

from score_based_model.mnist.utils import get_mnist_score_model, save_samples
from score_based_model.mnist.functions import marginal_prob_std, diffusion_coeff


def pc_sampler(score_model, marginal_prob_std, diffusion_coeff, batch_size=64, num_steps=500, snr=0.16, device="cuda", eps=1e-3):
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1.0, eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

            # The last step does not include any noise
        return x_mean


def sampling(ckpt_path, path_to_save, filename, device="cuda"):
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=device)

    score_model = get_mnist_score_model(ckpt_path, marginal_prob_std_fn, device=device)

    sample_batch_size = 64
    sampler = pc_sampler

    samples = sampler(
        score_model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        num_steps=500,
        snr=0.16,
        device=device,
    )

    save_samples(samples, sample_batch_size, path_to_save, filename)
