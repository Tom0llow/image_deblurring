from app.utils import create_results_dir, save_originals
from app.run import run

if __name__ == "__main__":
    create_results_dir(class_name="celebA")

    folder_to_save = "image/results/celebA"

    blur_image_path = "dataset/blurred_celebA/blur_images/000180.jpg"
    blur_kernel_path = "dataset/blurred_celebA/blur_kernels/000180.jpg"
    sharp_image_path = "dataset/blurred_celebA/sharp_images/000180.jpg"
    # save original images(kernels)
    fname = blur_image_path.split("/")[-1]
    save_originals(fname, paths=[blur_image_path, sharp_image_path, blur_kernel_path], path_to_save=folder_to_save)

    image_ckpt_path = "score_based_model/checkpoints/celebA/checkpoint.pth"
    kernel_ckpt_path = "score_based_model/checkpoints/blur_kernel/checkpoint.pth"

    run(
        path_to_save=folder_to_save,
        blur_image_path=blur_image_path,
        image_ckpt_path=image_ckpt_path,
        kernel_ckpt_path=kernel_ckpt_path,
        device="cuda",
    )
