from app.utils import create_results_dir, save_originals
from app.config import Config

if __name__ == "__main__":
    create_results_dir(class_name="celebA")

    folder_to_save = "image/results/celebA"

    blur_image_path = "dataset/blurred_celebA/blur_images/000010.jpg"
    blur_kernel_path = "dataset/blurred_celebA/blur_kernels/000010.jpg"
    sharp_image_path = "dataset/blurred_celebA/sharp_images/000010.jpg"
    # save original images(kernels)
    fname = blur_image_path.split("/")[-1]
    save_originals(fname, paths=[blur_image_path, sharp_image_path, blur_kernel_path], path_to_save=folder_to_save)

    image_ckpt_path = "score_based_model/checkpoints/celebA/checkpoint.pth"
    kernel_ckpt_path = "score_based_model/checkpoints/blur_kernel/checkpoint.pth"

    # load config
    params = Config.get_conf()
    if params["blind"] is True:
        from app.blind_deconvolution import run

        print("Blind Deconvolution")
        run(
            params,
            path_to_save=folder_to_save,
            blur_image_path=blur_image_path,
            image_ckpt_path=image_ckpt_path,
            kernel_ckpt_path=kernel_ckpt_path,
            device="cuda",
        )

    else:
        from app.deconvolution import run

        print("Deconvolution(use known kernel)")
        run(
            params,
            path_to_save=folder_to_save,
            blur_image_path=blur_image_path,
            kernel_image_path=blur_kernel_path,
            image_ckpt_path=image_ckpt_path,
            device="cuda",
        )
