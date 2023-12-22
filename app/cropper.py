import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


class Cropper:
    def __init__(self, image_path, size=(64, 64), path__to_save=None):
        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = Image.open(self.image_path)
        else:
            raise Exception("Not correct path to image.")
        self.size = size
        self.path_to_save = path__to_save
        self.result = None

    def cropper(self, save=False):
        x_center = self.original.size[0] // 2
        y_center = self.original.size[1] // 2
        half_short_side = min(x_center, y_center)
        x0 = x_center - half_short_side
        y0 = y_center - half_short_side
        x1 = x_center + half_short_side
        y1 = y_center + half_short_side

        cropped = self.original.crop((x0, y0, x1, y1))
        resized = cropped.resize(self.size)
        result = cv2.cvtColor(np.array(resized, dtype=np.float32), cv2.COLOR_BGR2RGB)

        self.result = result
        if save:
            self.save()

    def save(self):
        if self.path_to_save is None:
            raise Exception("Please create Trajectory instance with path_to_save")
        cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split("/")[-1]), self.result)


if __name__ == "__main__":
    folder = "./data/raw/celebA"
    folder_to_save = "./data/results_sharp/celebA"

    for path in tqdm(os.listdir(folder)):
        # print(path)
        Cropper(os.path.join(folder, path), path__to_save=folder_to_save, size=(256, 256)).cropper(save=True)
    print("Complete Cropped !")
