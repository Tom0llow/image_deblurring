import torch
from tqdm import tqdm

from app.model import E, normalize, get_score, clip_grad_norm_
from app.model import ImageLoss
from app.model import LangevinGD
from app.utils import save_estimateds, plot_ave_losses, plot_params


def optimize(blur_image, kernel_image, image_score_fn, alpha_, lambda_, eta_, fname, path_to_save, save_interval=100, num_steps=1000, num_scales=10000, batch_size=64, eps=1e-3, device="cuda"):
    # Initial samples
    image_init = torch.randn(num_scales, 3, 256, 256, device=device)

    # model
    blur_image = normalize(blur_image)
    model_i = ImageLoss(blur_image, image_init, device=device)
    del image_init
    torch.cuda.empty_cache()

    # optimizer
    optim_i = LangevinGD(model_i.parameters(), alpha_, lambda_, eta_, num_scales)

    timesteps = torch.linspace(1.0, eps, num_steps, device=device)
    ave_losses = []

    estimated_i = E(model_i.state_dict()["x_i"])
    with tqdm(timesteps) as tqdm_epoch:
        for i, t in enumerate(tqdm_epoch):
            ave_loss = 0.0

            # optimize image
            loss_i = model_i(kernel_image)

            if not torch.isnan(loss_i):
                with torch.no_grad():
                    image_score = get_score(model_i.state_dict()["x_i"], t, image_score_fn, num_scales, batch_size)
                ## langevin step
                optim_i.zero_grad(set_to_none=True)
                loss_i.backward()
                clip_grad_norm_(image_score, max_norm=10000, norm_type=2)
                estimated_i = optim_i.step(image_score)

                del image_score
                torch.cuda.empty_cache()

                ave_loss += loss_i
            loss_i.detach_()

            if ave_loss == 0.0:
                break
            ave_losses.append(ave_loss.detach().cpu().numpy())

            tqdm_epoch.set_description(f"Average Deblur Loss = {ave_loss:.5f}")

            # save
            if i % save_interval == 0:
                # save image and kernel
                save_estimateds(fname, path_to_save, estimated_i=normalize(estimated_i.detach().clone()))
                # plot each values
                plot_ave_losses(path_to_save, losses=ave_losses)
                plot_params("image", path_to_save, means=optim_i.param_means, vars=optim_i.param_vars, scores=optim_i.score_norms, grads=optim_i.grad_norms)

    return normalize(estimated_i.detach().clone())
