import os


def create_results_dir(path_to_save):
    try:
        os.mkdir(path_to_save)
    except FileExistsError:
        print(f"- {path_to_save} is already exist.")

    for sub_dir_name in ["sharp_images", "blur_images", "blur_kernels"]:
        try:
            sub_dir = os.path.join(path_to_save, sub_dir_name)
            os.mkdir(sub_dir)
        except FileExistsError:
            print(f"- {sub_dir} is already exist.")
