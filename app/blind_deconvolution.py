import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T

from app.models.functions import E, normalize, conv2D
from app.models.utils import get_score, clip_grad_norm_, EarlyStopping
from app.models.loss import ImageLoss, KernelLoss
from app.models.LangevinGD import LangevinGD
from app.utils import save_estimateds, plot_graphs


# Alternating Optimization
def optimize(blur_image, image_size, kernel_size, image_score_fn, kernel_score_fn, lambda_, eta_, fname, path_to_save, save_interval=100, num_steps=1000, num_scales=10000, batch_size=64, patience=100, eps=1e-3, device="cuda"):
    channel, h, w = image_size
    is_rgb = True if channel == 3 else False
    is_resize = True if kernel_size != (64, 64) else False

    # Initial samples
    image_init = torch.randn(num_scales, *image_size, device=device)
    kernel_init = torch.randn(num_scales, 1, 64, 64, device=device)

    # model
    blur_image = normalize(blur_image)
    model_i = ImageLoss(blur_image, image_init, is_rgb, device=device)
    model_k = KernelLoss(blur_image, kernel_init, kernel_size, is_resize, device=device)
    del image_init
    del kernel_init
    torch.cuda.empty_cache()

    # optimizer
    optim_i = LangevinGD(model_i.parameters(), lambda_, eta_, num_scales)
    optim_k = LangevinGD(model_k.parameters(), lambda_, eta_, num_scales)

    timesteps = torch.linspace(1.0, eps, num_steps, device=device)
    ave_losses = []
    image_grads = []
    kernel_grads = []
    estimated_i = E(model_i.state_dict()["x_i"]) if is_rgb else E(model_i.state_dict()["x_i"]).repeat(3, 1, 1)
    estimated_k = E(model_k.state_dict()["x_k"])
    if is_resize:
        F.resize(E(model_k.state_dict()["x_k"]), size=kernel_size, interpolation=T.InterpolationMode.BILINEAR)
    estimated_k.squeeze_()
    earlyStopping = EarlyStopping(fname, path_to_save, patience=patience, verbose=True)
    for i, t in enumerate(timesteps):
        ave_loss = 0.0

        # optimize image
        loss_i = model_i(estimated_k.detach().clone() * 255)

        with torch.no_grad():
            image_score = get_score(model_i.state_dict()["x_i"], t, image_score_fn, num_scales, batch_size)
        ## langevin step
        optim_i.zero_grad(set_to_none=True)
        loss_i.backward()
        estimated_i = optim_i.step(image_score)
        estimated_i = torch.clip(estimated_i, 0, 1)
        estimated_i = normalize(estimated_i)
        if not is_rgb:
            estimated_i = estimated_i.repeat(3, 1, 1)
        del image_score
        torch.cuda.empty_cache()

        ave_loss += loss_i
        loss_i.detach_()

        # optimize kernel
        loss_k = model_k(estimated_i.detach().clone())
        with torch.no_grad():
            kernel_score = get_score(model_k.state_dict()["x_k"], t, kernel_score_fn, num_scales, batch_size)
        ## langevin step
        optim_k.zero_grad(set_to_none=True)
        loss_k.backward()
        estimated_k = optim_k.step(kernel_score)
        if is_resize:
            estimated_k = F.resize(estimated_k, size=kernel_size, interpolation=T.InterpolationMode.BILINEAR)
        estimated_k = torch.clip(estimated_k, 0, 1)
        estimated_k = normalize(estimated_k)
        estimated_k.squeeze_()
        del kernel_score
        torch.cuda.empty_cache()

        ave_loss += loss_k
        loss_k.detach_()

        ave_loss /= 2
        ave_losses.append(ave_loss.detach().cpu().numpy())
        image_grad_norm = torch.norm(optim_i.param_groups[0]["params"][0].grad)
        image_grads.append(image_grad_norm.detach().cpu().numpy())
        kernel_grad_norm = torch.norm(optim_k.param_groups[0]["params"][0].grad)
        kernel_grads.append(kernel_grad_norm.detach().cpu().numpy())

        if i % save_interval == 0:
            plot_graphs(fname, path_to_save, losses=ave_losses, image_grads=image_grads, kernel_grads=kernel_grads)
        # save best estimateds
        estimated_b = normalize(conv2D(estimated_i, estimated_k))
        earlyStopping(i, ave_loss, estimated_i=estimated_i.detach().clone(), estimated_k=estimated_k.detach().clone(), estimated_b=estimated_b.detach().clone())
        if earlyStopping.early_stop:
            print("Early Stopping!")
            break
