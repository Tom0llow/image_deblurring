import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import cv2
import os

from generate_trajectory import Trajectory


class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory = Trajectory(canvas=canvas, expl=0.005).fit(show=False, save=False)
        else:
            self.trajectory = trajectory.x
        if fraction is None:
            self.fraction = [1 / 100, 1 / 10, 1 / 2, 1]
        else:
            self.fraction = fraction
        self.PSFnumber = len(self.fraction)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self):
        PSF = np.zeros(self.canvas)

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                # print(j, t)
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                M2 = int(m2)
                m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                M1 = int(m1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(self.trajectory[t].real - m2, self.trajectory[t].imag - m1)
                PSF[m1, M2] += t_proportion * triangle_fun_prod(self.trajectory[t].real - M2, self.trajectory[t].imag - m1)
                PSF[M1, m2] += t_proportion * triangle_fun_prod(self.trajectory[t].real - m2, self.trajectory[t].imag - M1)
                PSF[M1, M2] += t_proportion * triangle_fun_prod(self.trajectory[t].real - M2, self.trajectory[t].imag - M1)

            self.PSFs.append(PSF / (self.iters))

        return self.PSFs


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
