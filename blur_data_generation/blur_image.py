import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image

from generate_PSF import PSF, scale


class BlurImage(object):
    def __init__(self, image_path, PSFs=None, part=None, path_to_save=None, is_rgb=True, device="cuda"):
        """
        :param image_path: path to square image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        self.is_rgb = is_rgb
        self.device = device

        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = read_image(self.image_path).to(device=self.device)
            self.original = self.original if self.is_rgb else self.original.repeat(3, 1, 1)
            self.shape = self.original.size()
            if self.shape[1] != self.shape[2]:
                raise Exception("We support only square images yet.")
        else:
            raise Exception("Not correct path to image.")
        self.path_to_save = path_to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[1]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[1], path_to_save=os.path.join(self.path_to_save, "PSFs.png")).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []

        self.toTensor = transforms.ToTensor()

    def normalize(self, x):
        size = x.size()

        x = x.to(torch.float)
        x = x.view(x.size(0), -1)
        x = x - x.min(1, keepdim=True)[0]
        x = x / x.max(1, keepdim=True)[0]
        x = x.view(*size)
        return x

    def tensor_to_ndarray(self, x):
        if self.is_rgb:
            x = x.permute(1, 2, 0)
            x = x.cpu().detach().numpy()
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        else:
            x = 0.114 * x[0, :, :] + 0.587 * x[1, :, :] + 0.299 * x[2, :, :]
            x = x.cpu().detach().numpy()
        return x

    def blur_image(self, save=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]

        channel, yN, xN = self.shape
        key, kex = self.PSFs[0].shape
        delta = yN - key
        assert delta >= 0, "resolution of image should be higher than kernel"
        result = []
        if len(psf) > 1:
            for p in psf:
                p = scale(p, 0, 255)
                p = self.toTensor(p).squeeze().to(device=self.device)

                tmp = torch.nn.functional.pad(p, (delta // 2, delta // 2, delta // 2, delta // 2), "constant")
                tmp = self.normalize(tmp).nan_to_num()
                tmp = torch.eye(3)[..., None, None].to(device=self.device) * tmp[None, None, ...]

                blured = self.normalize(self.original).unsqueeze(0)
                blured = torch.nn.functional.conv2d(blured, tmp, padding="same")
                blured = self.normalize(blured)

                blured.squeeze_()
                blured = self.tensor_to_ndarray(blured)
                result.append(blured)
        else:
            psf = psf[0]
            psf = scale(psf, 0, 255)
            psf = self.toTensor(psf).squeeze().to(device=self.device)

            tmp = torch.nn.functional.pad(psf, (delta // 2, delta // 2, delta // 2, delta // 2), "constant")
            tmp = self.normalize(tmp).nan_to_num()
            tmp = torch.eye(3)[..., None, None].to(device=self.device) * tmp[None, None, ...]

            blured = self.normalize(self.original).unsqueeze(0)
            blured = torch.nn.functional.conv2d(blured, tmp, padding="same")
            blured = self.normalize(blured)

            blured.squeeze_()
            blured = self.tensor_to_ndarray(blured)
            result.append(blured)
        self.result = result

        if save:
            self.save()

    def save(self):
        self.original = self.tensor_to_ndarray(self.original)

        psf = None
        psf = self.PSFs[-1] if self.part is None else self.PSFs[self.part]

        if len(self.result) == 0:
            raise Exception("Please run blur_image() method first.")
        else:
            if self.path_to_save is None:
                raise Exception("Please create Trajectory instance with path_to_save")

            cv2.imwrite(os.path.join(self.path_to_save + "/sharp_images", self.image_path.split("/")[-1]), self.original)  # sharp_image
            cv2.imwrite(os.path.join(self.path_to_save + "/blur_kernels", self.image_path.split("/")[-1]), scale(psf, 0, 255))  # psf
            cv2.imwrite(os.path.join(self.path_to_save + "/blur_images", self.image_path.split("/")[-1]), self.result[0] * 255)  # blur_iamge
