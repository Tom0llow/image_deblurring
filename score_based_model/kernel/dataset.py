import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from PIL import Image


class ImageTransform:
    def __init__(self):
        self.data_transform = transforms.ToTensor()

    def __call__(self, img):
        return self.data_transform(img)


class Kernel_Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("L")
        img_label = "kernel"
        img_transformed = self.transform(img)
        return img_transformed, img_label, img_path
