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

    for sub_dir_name in ["blur_images", "estimated_images", "estimated_kernels", "original_images", "original_kernels", "outputs"]:
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


def save_estimateds(fname, path_to_save, estimated_i, estimated_k=None):
    ## image
    estimated_i = estimated_i.permute(1, 2, 0)
    estimated_i = estimated_i.cpu().detach().numpy()
    estimated_i = cv2.cvtColor(estimated_i, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path_to_save + "/estimated_images", fname), estimated_i * 255)

    ## kernel
    if estimated_k is not None:
        estimated_k = estimated_k.permute(1, 2, 0).squeeze()
        estimated_k = estimated_k.cpu().detach().numpy()
        cv2.imwrite(os.path.join(path_to_save + "/estimated_kernels", fname), estimated_k * 255)
        print("Saved estimated image and kernel.")

    print("Saved estimated image")


def plot_graphs(path_to_save, losses, image_grads, kernel_grads=None):
    if kernel_grads is not None:
        # blind deconvolution
        plt.figure(figsize=(30, 8))
        # plot losses
        plt.subplot(131, title="loss")
        plt.plot(np.arange(len(losses)), losses, marker="o", color="b")
        plt.xlabel("time step")
        plt.grid()
        # plot image grads
        plt.subplot(132, title="image gradient")
        plt.plot(np.arange(len(image_grads)), image_grads, marker="o", color="m")
        plt.xlabel("time step")
        plt.grid()
        # plot kernel grads
        plt.subplot(133, title="kernel gradient")
        plt.plot(np.arange(len(kernel_grads)), kernel_grads, marker="o", color="r")
        plt.xlabel("time step")
        plt.grid()
    else:
        # non-blind deconvolution
        plt.figure(figsize=(20, 8))
        # plot losses
        plt.subplot(121, title="loss")
        plt.plot(np.arange(len(losses)), losses, marker="o", color="b")
        plt.xlabel("time step")
        plt.grid()
        # plot image grads
        plt.subplot(122, title="image gradient")
        plt.plot(np.arange(len(image_grads)), image_grads, marker="o", color="m")
        plt.xlabel("time step")
        plt.grid()

    # set options
    plt.tight_layout()
    # save
    plt.savefig(os.path.join(path_to_save + "/outputs", "graphs.png"))
    plt.close()
    plt.clf()
