import os
from tqdm import tqdm

from cropper import Cropper

if __name__ == "__main__":
    folder = "./data/raw/celebA"
    folder_to_save = "./data/results_sharp/celebahq_256"

    for path in tqdm(os.listdir(folder)):
        # print(path)
        Cropper(os.path.join(folder, path), path__to_save=folder_to_save, size=(256, 256)).cropper(save=True)
    print("Complete Cropped !")
