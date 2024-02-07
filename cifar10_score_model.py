from score_based_model.cifar10.sampling import run


if __name__ == "__main__":
    image_ckpt_path = "score_based_model/checkpoints/cifar10/checkpoint.pth"
    path_to_save = "image/sample"
    filename = "cifar10_ncsnpp_continuous.jpg"

    run(image_ckpt_path, path_to_save, filename, device="cuda")
