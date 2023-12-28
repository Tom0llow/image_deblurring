import torch
from torch.nn import functional as F
import math


def E(x):
    # expected value
    return x.float().mean(dim=0)


def normalize(x):
    # min-max normalize
    size = x.size()

    x = x.to(torch.float)
    x = x.view(x.size(0), -1)
    x = x - x.min(1, keepdim=True)[0]
    x = x / x.max(1, keepdim=True)[0]
    x = x.view(*size)
    return x


def conv2D(image, kernel):
    # blur
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
