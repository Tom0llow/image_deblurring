import os
import cv2
import numpy as np
from generate_PSF import PSF
from generate_trajectory import Trajectory
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image

from PSF import scale


def create_results_dir(path_to_save):
    for sub_dir_name in ["sharp_images", "blur_images", "blur_kernels"]:
        try:
            sub_dir = os.path.join(path_to_save, sub_dir_name)
            os.mkdir(sub_dir)
        except FileExistsError:
            print(f"- {sub_dir} is already exist.")


class BlurImage(object):
    def __init__(self, image_path, PSFs=None, part=None, path_to_save=None, device="cuda"):
        """
        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        self.device = device

        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = read_image(self.image_path).to(device=self.device)
            self.shape = self.original.size()
            if len(self.shape) < 3:
                raise Exception("We support only RGB images yet.")
            elif self.shape[1] != self.shape[2]:
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
        x = x.permute(1, 2, 0)
        x = x.cpu().detach().numpy()
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
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


if __name__ == "__main__":
    folder = "./dataset/celebA/test"
    folder_to_save = "./dataset/blurred_celebA"
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]

    create_results_dir(path_to_save=folder_to_save)
    for path in tqdm(os.listdir(folder)):
        # print(path)
        trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
        psf = PSF(canvas=64, trajectory=trajectory).fit()
        BlurImage(os.path.join(folder, path), PSFs=psf, path_to_save=folder_to_save, part=np.random.choice([1, 2, 3])).blur_image(save=True)

    print("Complete generated blur image !")
