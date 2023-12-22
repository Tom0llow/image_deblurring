import cv2
import numpy as np
from generate_PSF import PSF
from generate_trajectory import Trajectory
from tqdm import tqdm


def scale(x, a, b):
    if x.min() == x.max():
        return np.zeros_like(x)
    scaled = (b - a) * (x - x.min()) / (x.max() - x.min()) + a
    return scaled


if __name__ == "__main__":
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]

    N = 80000  # data_num
    for i in range(N):
        trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
        psf = PSF(canvas=64, trajectory=trajectory).fit()

        part = np.random.choice([1, 2, 3])
        cv2.imwrite(f"./data/RandomMotionBlur/psf_{f'{i:05}'}.png", scale(psf[part], 0, 255))
    print("Complete generated PSF !")
