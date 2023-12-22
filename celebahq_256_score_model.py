from score_based_model.celebahq_256.sampling import run


if __name__ == "__main__":
    image_ckpt_path = "score_based_model/checkpoints/celebA/checkpoint.pth"
    path_to_save = "image/sample"
    filename = "celebahq_256_ncsnpp_continuous.jpg"

    run(image_ckpt_path, path_to_save, filename, device="cuda")
