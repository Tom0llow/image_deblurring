import torch
from torch.nn import functional as F


def E(x):
    # expected value
    return x.float().mean(dim=0)


def normalize(x):
    # min-max normalize
    size = x.size()

    x = x.to(torch.float)
    x = x.view(x.size(0), -1)
    x = x - x.min(1, keepdim=True)[0]
    x = x / (x.max(1, keepdim=True)[0] + 1e-12)
    x = x.view(*size)
    return x


def conv2D(image, kernel):
    # blur
    channel, yN, xN = image.size()
    key, kex = kernel.size()
    delta = yN - key

    tmp = F.pad(kernel, (delta // 2, delta // 2, delta // 2, delta // 2), "constant")
    tmp = torch.eye(3)[..., None, None].to(tmp.device) * tmp[None, None, ...]

    blurred = image.unsqueeze(0)
    blurred = F.conv2d(blurred, tmp, padding="same")
    blurred.squeeze_()

    return blurred
