import torch
from tqdm import tqdm

from app.models.functions import E, normalize
from app.models.utils import get_score, clip_grad_norm_
from app.models.model import ImageLoss
from app.models.LangevinGD import LangevinGD
from app.utils import save_estimateds, plot_graphs


def optimize(blur_image, blur_kernel, image_size, image_score_fn, lambda_, eta_, fname, path_to_save, save_interval=100, num_steps=1000, num_scales=10000, batch_size=64, eps=1e-3, device="cuda"):
    channel, h, w = image_size
    is_rgb = True if channel == 3 else False

    # Initial samples
    image_init = torch.randn(num_scales, *image_size, device=device)

    # model
    blur_image = normalize(blur_image)
    model_i = ImageLoss(blur_image, image_init, is_rgb, device=device)
    del image_init
    torch.cuda.empty_cache()

    # optimizer
    optim_i = LangevinGD(model_i.parameters(), lambda_, eta_, num_scales)

    timesteps = torch.linspace(1.0, eps, num_steps, device=device)
    ave_losses = []
    image_grads = []

    with tqdm(timesteps) as tqdm_epoch:
        for i, t in enumerate(tqdm_epoch):
            ave_loss = 0.0

            # optimize image
            loss_i = model_i(blur_kernel)

            with torch.no_grad():
                image_score = get_score(model_i.state_dict()["x_i"], t, image_score_fn, num_scales, batch_size)
            ## langevin step
            optim_i.zero_grad(set_to_none=True)
            loss_i.backward()
            estimated_i = optim_i.step(image_score)
            if not is_rgb:
                estimated_i = estimated_i.repeat(3, 1, 1)
            ave_loss += loss_i

            del image_score
            torch.cuda.empty_cache()
            # loss_i.detach_()

            ave_losses.append(ave_loss.detach().cpu().numpy())
            image_grad_norm = torch.norm(optim_i.param_groups[0]["params"][0].grad)
            image_grads.append(image_grad_norm.detach().cpu().numpy())

            tqdm_epoch.set_description(f"Loss:{ave_loss:5f}, Image Grad Norm:{image_grad_norm:5f}")

            # save
            if i % save_interval == 0:
                save_estimateds(fname, path_to_save, estimated_i=normalize(estimated_i.detach().clone()))
                plot_graphs(path_to_save, losses=ave_losses, image_grads=image_grads)

    return normalize(estimated_i.detach().clone())
