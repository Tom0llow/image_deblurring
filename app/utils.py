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


def plot_ave_losses(path_to_save, losses):
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(losses)), losses, marker="o")
    plt.xlabel("time step")
    plt.title("Average Deblurring Loss")
    plt.grid()
    plt.savefig(os.path.join(path_to_save + "/outputs", "ave_losses.png"))
    plt.close()
    plt.clf()


def plot_params(class_name, path_to_save, params, scores, grads):
    # init
    plt.figure(figsize=(24, 8))
    # plot params
    plt.subplot(131, title="param mean")
    plt.plot(np.arange(len(params)), params, marker="o", color="b")
    plt.xlabel("time step")
    plt.grid()
    # plot scores
    plt.subplot(132, title="score norm")
    plt.plot(np.arange(len(scores)), scores, marker="^", color="g")
    plt.xlabel("time step")
    plt.grid()
    # plot grads
    plt.subplot(133, title="grad norm")
    plt.plot(np.arange(len(grads)), grads, marker="x", color="r")
    plt.xlabel("time step")
    plt.grid()
    # set options
    plt.suptitle(f"[{class_name}]")
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save + "/outputs", f"{class_name}_params.png"))
    plt.close()
    plt.clf()
