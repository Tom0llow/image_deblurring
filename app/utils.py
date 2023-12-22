import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_results_dir(class_name, results_path="image/results"):
    try:
        class_dir = os.path.join(results_path, class_name)
        os.mkdir(class_dir)
    except FileExistsError:
        print(f"- {class_dir} is already exist.")

    for sub_dir_name in ["blur_images", "estimated_images", "estimated_kernels", "original_images", "original_kernels"]:
        try:
            sub_dir = os.path.join(class_dir, sub_dir_name)
            os.mkdir(sub_dir)
        except FileExistsError:
            print(f"- {sub_dir} is already exist.")


def save_originals(fname, paths, path_to_save):
    sub_dir_names = ["blur_images", "original_images", "original_kernels"]
    for i in range(len(sub_dir_names)):
        sub_dir = os.path.join(path_to_save, sub_dir_names[i])

        src = paths[i]
        dst = os.path.join(sub_dir, fname)
        shutil.copyfile(src, dst)

    print("Saved original image and kernel.")


def save_estimateds(fname, path_to_save, estimated_i, estimated_k):
    ## image
    estimated_i = estimated_i.permute(1, 2, 0)
    estimated_i = estimated_i.cpu().detach().numpy()
    estimated_i = cv2.cvtColor(estimated_i, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path_to_save + "/estimated_images", fname), estimated_i * 255)

    ## kernel
    estimated_k = estimated_k.permute(1, 2, 0).squeeze()
    estimated_k = estimated_k.cpu().detach().numpy()
    cv2.imwrite(os.path.join(path_to_save + "/estimated_kernels", fname), estimated_k * 255)

    print("Saved estimated image and kernel.")


def plot_grads(path_to_save, norms_i, norms_k):
    def plot_grad(norms):
        x = np.arange(len(norms))
        y = norms
        plt.plot(x, y, marker="o")
        for i, v in enumerate(norms):
            plt.text(x[i], y[i], f"{v:.3f}")
        plt.xlabel("time step")
        plt.ylabel("grad norm")

    ## image
    plt.figure()
    plot_grad(norms_i)
    plt.title("image grad")
    plt.savefig(os.path.join(path_to_save + "/outputs", "image_grad.png"))
    plt.clf()

    ## kernel
    plt.figure()
    plot_grad(norms_k)
    plt.title("kernel grad")
    plt.savefig(os.path.join(path_to_save + "/outputs", "kernel_grad.png"))
    plt.clf()


def plot_ave_losses(path_to_save, losses):
    plt.figure()
    x = np.arange(len(losses))
    y = losses
    plt.plot(x, y, marker="o")
    for i, v in enumerate(losses):
        plt.text(x[i], y[i], f"{v:.3f}")
    plt.xlabel("time step")
    plt.ylabel("ave loss")
    plt.title("deblurring loss")
    plt.savefig(os.path.join(path_to_save + "/outputs", "ave_losses.png"))
    plt.clf()
