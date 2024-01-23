import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T

from app.models.functions import E, normalize, conv2D


class DeblurLoss(torch.nn.Module):
    def __init__(self):
        super(DeblurLoss, self).__init__()

    def forward(self):
        pass


class ImageLoss(DeblurLoss):
    def __init__(self, blur_image, image_init, is_rgb=True, device="cuda"):
        super().__init__()
        self.is_rgb = is_rgb
        self.device = device
        self.blur_image = blur_image.to(self.device)
        self.x_i = torch.nn.Parameter(image_init.requires_grad_(True).to(device))

    def forward(self, kernel):
        real_b = self.blur_image
        estimated_i = normalize(E(self.x_i)) if self.is_rgb else normalize(E(self.x_i).repeat(3, 1, 1))
        kernel = normalize(kernel.squeeze().to(self.device))

        estimated_b = conv2D(estimated_i, kernel)
        return torch.norm(real_b - estimated_b) ** 2


class KernelLoss(DeblurLoss):
    def __init__(self, blur_image, kernel_init, kernel_size=(64, 64), is_resize=False, device="cuda"):
        super().__init__()
        self.kernel_size = kernel_size
        self.is_resize = is_resize
        self.device = device
        self.blur_image = blur_image.to(device)
        self.x_k = torch.nn.Parameter(kernel_init.requires_grad_(True).to(device))

    def forward(self, image):
        real_b = self.blur_image
        image = normalize(image.to(self.device))
        estimated_k = normalize(E(self.x_k))
        if self.is_resize:
            estimated_k = F.resize(estimated_k, size=self.kernel_size, interpolation=T.InterpolationMode.NEAREST)
        estimated_k.squeeze_()

        estimated_b = conv2D(image, estimated_k)
        return torch.norm(real_b - estimated_b) ** 2
