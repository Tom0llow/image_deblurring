import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage


def calc(o_folder, e_folder):
    psnr = 0
    ssim = 0
    num = 0
    for o_name, e_name in zip(os.listdir(o_folder), os.listdir(e_folder)):
        o_path = os.path.join(o_folder, o_name)
        e_path = os.path.join(e_folder, e_name)

        original = cv2.imread(o_path, cv2.IMREAD_GRAYSCALE)
        estimated = cv2.imread(e_path, cv2.IMREAD_GRAYSCALE)

        psnr += skimage.metrics.peak_signal_noise_ratio(original, estimated)
        ssim += skimage.metrics.structural_similarity(original, estimated)
        num += 1
        break

    psnr /= num
    ssim /= num
    return psnr, ssim


def plot_graphs(fname, path_to_save, psnrs, ssims):
    plt.figure(figsize=(20, 8))
    # plot psnr
    plt.subplot(121)
    plt.plot(np.arange(len(psnrs)), psnrs, marker="o", color="b")
    plt.xticks(np.arange(len(psnrs)))
    plt.xlabel("% noise")
    plt.ylabel("PSNR")
    plt.grid()
    # plot ssim
    plt.subplot(122)
    plt.plot(np.arange(len(ssims)), ssims, marker="o", color="g")
    plt.xticks(np.arange(len(ssims)))
    plt.xlabel("% noise")
    plt.ylabel("SSIM")
    plt.grid()
    # set options
    plt.tight_layout()
    # save
    plt.savefig(os.path.join(path_to_save, f"{fname}_eval.png"))
    plt.close()
    plt.clf()


def eval(class_name):
    path_to_save = f"./eval/{class_name}"
    # make dir
    try:
        os.mkdir(path_to_save)
    except FileExistsError:
        print(f"{path_to_save} is already exist.")

    psnrs = []
    ssims = []
    for i in range(0, 10 + 1):
        o_folder = f"./results/{class_name} ({i}_perc_noise)/original_images"
        e_folder = f"./results/{class_name} ({i}_perc_noise)/estimated_images"

        psnr, ssim = calc(o_folder, e_folder)
        psnrs.append(psnr)
        ssims.append(ssim)

    plot_graphs(class_name, path_to_save, psnrs, ssims)


if __name__ == "__main__":
    eval(class_name="deconv_mnist")
