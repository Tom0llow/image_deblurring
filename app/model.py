import torch
from torch.nn import functional as F
import math
from tqdm import tqdm

from app.utils import save_estimateds, plot_ave_losses, plot_params


def E(x):
    return x.float().mean(dim=0)


def normalize(x):
    size = x.size()

    x = x.to(torch.float)
    x = x.view(x.size(0), -1)
    x = x - x.min(1, keepdim=True)[0]
    x = x / x.max(1, keepdim=True)[0]
    x = x.view(*size)
    return x


def conv2D(image, kernel):
    channel, yN, xN = image.size()
    key, kex = kernel.size()
    delta = yN - key

    tmp = F.pad(kernel, (delta // 2, delta // 2, delta // 2, delta // 2), "constant")
    tmp = normalize(tmp).nan_to_num()
    tmp = torch.eye(3)[..., None, None].to(tmp.device) * tmp[None, None, ...]

    blured = normalize(image).unsqueeze(0)
    blured = F.conv2d(blured, tmp, padding="same")
    blured = normalize(blured)

    return blured


def get_score(x, t, score_fn, num_scales, batch_size):
    def get_batch_score(x_batch, t, score_fn, batch_size):
        vec_t = torch.ones(batch_size, device=t.device) * t
        vec_t = vec_t.to(x_batch.device)
        return score_fn(x_batch, vec_t)

    score = torch.empty(1, *x.size()[1:], device=x.device)

    N = math.ceil(num_scales // batch_size) - 1
    if N == 0:
        score = get_batch_score(x, t, score_fn, batch_size)
        return score

    idx = list(range(0, num_scales, batch_size))
    for i in range(N):
        x_batch = x[idx[i] : idx[i + 1]]
        batch_score = get_batch_score(x_batch, t, score_fn, batch_size)
        score = torch.cat((score, batch_score), 0)
    score = torch.cat((score, x[idx[i + 1] :]), 0)

    return score[1:]


class DeblurLoss(torch.nn.Module):
    def __init__(self):
        super(DeblurLoss, self).__init__()

    def forward(self):
        pass


class ImageLoss(DeblurLoss):
    def __init__(self, blur_image, image_init, device="cuda"):
        super().__init__()
        self.device = device
        self.blur_image = blur_image.to(self.device)
        self.x_i = torch.nn.Parameter(image_init.requires_grad_(True).to(device))

    def forward(self, kernel):
        real_b = self.blur_image
        estimated_i = normalize(E(self.x_i))
        kernel = normalize(kernel.squeeze_().to(self.device))

        estimated_b = conv2D(estimated_i, kernel)
        return torch.norm(real_b - estimated_b) ** 2


class KernelLoss(DeblurLoss):
    def __init__(self, blur_image, kernel_init, device="cuda"):
        super().__init__()
        self.device = device
        self.blur_image = blur_image.to(device)
        self.x_k = torch.nn.Parameter(kernel_init.requires_grad_(True).to(device))

    def forward(self, image):
        real_b = self.blur_image
        image = normalize(image.to(self.device))
        estimated_k = normalize(E(self.x_k).squeeze_())

        estimated_b = conv2D(image, estimated_k)
        return torch.norm(real_b - estimated_b) ** 2


# mean-field Langevin Dynamics
class LangevinGD(torch.optim.Optimizer):
    def __init__(self, params, alpha_, lambda_, eta_, m, snr):
        defaults = dict(alpha_=alpha_, lambda_=lambda_, eta_=eta_, m=m, snr=snr)
        super(LangevinGD, self).__init__(params, defaults)
        self.param_means = []
        self.score_norms = []
        self.grad_norms = []

    def step(self, score_fn):
        assert len(self.param_groups) == 1
        for group in self.param_groups:
            for p in group["params"]:
                alpha_ = group["alpha_"]
                lambda_ = group["lambda_"]
                lr = group["m"] * group["eta_"]
                snr = group["snr"]

                noise = torch.randn_like(p.data)

                grad = p.grad.data
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                score_norm = torch.norm(score_fn.reshape(score_fn.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()

                grad_stepsize = (noise_norm / grad_norm) ** 2 * 2 * lr * alpha_ * 0.5
                score_stepsize = (snr * noise_norm / score_norm) ** 2 * 2 * lr * lambda_

                p_mean = p.data + (1 - score_stepsize) * p.data - score_stepsize * score_fn - grad_stepsize * grad
                p.data = p_mean + math.sqrt(2 * score_stepsize) * noise

                self.param_means.append(torch.mean(p.data.detach().clone()).cpu().numpy())
                self.score_norms.append(score_norm.detach().cpu().numpy())
                self.grad_norms.append(grad_norm.detach().cpu().numpy())
                # The last step does not include any noise
                return E(p_mean)


# Alternating Optimization
def optimize(
    blur_image,
    image_score_fn,
    kernel_score_fn,
    marginal_prob_std,
    alpha_,
    lambda_,
    eta_,
    fname,
    path_to_save,
    save_interval=100,
    num_steps=1000,
    num_scales=10000,
    batch_size=64,
    eps=1e-3,
    device="cuda",
):
    # Initial samples
    t = torch.ones(num_scales, device=device)
    image_init = torch.randn(num_scales, 3, 256, 256, device=device) * 50  # celebahq_256_ncsnpp_continuous
    kernel_init = torch.randn(num_scales, 1, 64, 64, device=device) * marginal_prob_std(t)[:, None, None, None]

    # model
    blur_image = normalize(blur_image)
    model_i = ImageLoss(blur_image, image_init, device=device)
    model_k = KernelLoss(blur_image, kernel_init, device=device)
    del image_init
    del kernel_init
    torch.cuda.empty_cache()

    # optimizer
    optim_i = LangevinGD(model_i.parameters(), alpha_, lambda_, eta_, num_scales, snr=0.16)
    optim_k = LangevinGD(model_k.parameters(), alpha_, lambda_, eta_, num_scales, snr=0.16)

    timesteps = torch.linspace(1.0, eps, num_steps, device=device)
    ave_losses = []

    estimated_i = E(model_i.state_dict()["x_i"])
    estimated_k = E(model_k.state_dict()["x_k"])
    with tqdm(timesteps) as tqdm_epoch:
        for i, t in enumerate(tqdm_epoch):
            ave_loss = 0.0

            # optimize image
            loss_i = model_i(estimated_k.detach().clone())
            if not torch.isnan(loss_i):
                with torch.no_grad():
                    image_score = get_score(model_i.state_dict()["x_i"], t, image_score_fn, num_scales, batch_size)
                ## langevin step
                optim_i.zero_grad(set_to_none=True)
                loss_i.backward()
                estimated_i = optim_i.step(image_score)

                del image_score
                torch.cuda.empty_cache()

                ave_loss += loss_i
            loss_i.detach_()

            # optimize kernel
            loss_k = model_k(estimated_i.detach().clone())
            if not torch.isnan(loss_k):
                with torch.no_grad():
                    kernel_score = get_score(model_k.state_dict()["x_k"], t, kernel_score_fn, num_scales, batch_size)
                ## langevin step
                optim_k.zero_grad(set_to_none=True)
                loss_k.backward()
                estimated_k = optim_k.step(kernel_score)

                del kernel_score
                torch.cuda.empty_cache()

                ave_loss += loss_k
            loss_k.detach_()

            if ave_loss == 0.0:
                break
            else:
                ave_loss /= 2
            ave_losses.append(ave_loss.detach().cpu().numpy())

            tqdm_epoch.set_description(f"Average Deblur Loss = {ave_loss:.5f}")

            # save
            if i % save_interval == 0:
                # save image and kernel
                save_estimateds(fname, path_to_save, estimated_i=normalize(estimated_i.detach().clone()), estimated_k=normalize(estimated_k.detach().clone()))
                # plot each values
                plot_ave_losses(path_to_save, losses=ave_losses)
                plot_params('image', path_to_save, params=optim_i.param_means, scores=optim_i.score_norms, grads=optim_i.grad_norms)
                plot_params('kernel', path_to_save, params=optim_k.param_means, scores=optim_k.score_norms, grads=optim_k.grad_norms)

    return normalize(estimated_i.detach().clone()), normalize(estimated_k.detach().clone())
