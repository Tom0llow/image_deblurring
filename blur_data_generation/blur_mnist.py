import os
import numpy as np
from tqdm import tqdm

from generate_trajectory import Trajectory
from generate_PSF import PSF
from blur_image import BlurImage
from utils import create_results_dir


if __name__ == "__main__":
    folder = "./dataset/mnist/test"
    folder_to_save = "./dataset/blured_mnist"
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]

    create_results_dir(path_to_save=folder_to_save)
    for path in tqdm(os.listdir(folder)):
        # print(path)
        trajectory = Trajectory(canvas=28, max_len=28, expl=np.random.choice(params)).fit()
        psf = PSF(canvas=28, trajectory=trajectory).fit()
        BlurImage(os.path.join(folder, path), PSFs=psf, path_to_save=folder_to_save, part=np.random.choice([1, 2, 3]), is_rgb=False).blur_image(save=True)

    print("Complete generated blur image !")
