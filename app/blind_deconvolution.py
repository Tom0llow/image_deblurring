from app.blind_optimizer import optimize
from app.utils import save_estimateds


# Blind Deconvolution
def run(params, fname, path_to_save, blur_image, image_score_model, kernel_score_model, image_size, kernel_size, device="cuda"):
    estimated_i, estimated_k = optimize(
        blur_image,
        image_size,
        kernel_size,
        image_score_model,
        kernel_score_model,
        lambda_=params["lambda_"],
        eta_=params["eta_"],
        fname=fname,
        path_to_save=path_to_save,
        save_interval=params["save_interval"],
        num_steps=params["num_steps"],
        num_scales=params["num_scales"],
        batch_size=params["batch_size"],
        device=device,
    )

    # save
    save_estimateds(
        fname=fname,
        path_to_save=path_to_save,
        estimated_i=estimated_i,
        estimated_k=estimated_k,
    )
