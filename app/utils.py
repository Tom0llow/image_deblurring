import os
from contextlib import redirect_stdout
import cv2
import matplotlib.pyplot as plt
import numpy as np
from functools import partialmethod


def run(processes):
    if processes is None:
        return

    if len(processes) > 1:
        p1 = processes[0]
        p1.start()
        with redirect_stdout(open(os.devnull, "w")):
            for p in processes[1:]:
                p.start()
            for p in processes[1:]:
                p.join()
        p1.join()
    else:
        p = processes[0]
        p.start()
        p.join()


def create_results_dir(class_name, results_path):
    try:
        class_dir = os.path.join(results_path, class_name)
        os.mkdir(class_dir)
    except FileExistsError:
        print(f"- {class_dir} is already exist.")

    for sub_dir_name in ["original_blur_images", "original_images", "original_kernels", "estimated_images", "estimated_kernels", "estimated_blur_images", "outputs"]:
        try:
            sub_dir = os.path.join(class_dir, sub_dir_name)
            os.mkdir(sub_dir)
        except FileExistsError:
            print(f"- {sub_dir} is already exist.")

    return class_dir


def save(fname, class_name, path_to_save, image, kernel_image=None, blur_image=None):
    # image
    i = image.permute(1, 2, 0)
    i = i.cpu().detach().numpy()
    i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(path_to_save + f"/{class_name}_images", fname), i * 255)

    # kernel
    if kernel_image is not None:
        k = kernel_image.cpu().detach().numpy()
        cv2.imwrite(os.path.join(path_to_save + f"/{class_name}_kernels", fname), k * 255)

    # blur image
    if blur_image is not None:
        b = blur_image.permute(1, 2, 0)
        b = b.cpu().detach().numpy()
        b = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path_to_save + f"/{class_name}_blur_images", fname), b * 255)


def save_originals(fname, path_to_save, sharp_image, kernel_image, blur_image):
    save(
        fname,
        class_name="original",
        path_to_save=path_to_save,
        image=sharp_image,
        kernel_image=kernel_image,
        blur_image=blur_image,
    )
    print("Saved original image and kernel.")


def save_estimateds(fname, path_to_save, estimated_i, estimated_k=None, estimated_b=None):
    save(
        fname,
        class_name="estimated",
        path_to_save=path_to_save,
        image=estimated_i,
        kernel_image=estimated_k,
        blur_image=estimated_b,
    )


def plot_graphs(fname, path_to_save, losses, image_grads, kernel_grads=None):
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
    plt.savefig(os.path.join(path_to_save + "/outputs", "graphs_" + fname))
    plt.close()
    plt.clf()
