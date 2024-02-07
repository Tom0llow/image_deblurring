from score_based_model.score_sde_pytorch.losses import get_optimizer
from score_based_model.score_sde_pytorch.models import utils as mutils
from score_based_model.score_sde_pytorch.models.ema import ExponentialMovingAverage
from score_based_model.score_sde_pytorch.utils import restore_checkpoint
from score_based_model.score_sde_pytorch.models import ncsnpp
from score_based_model.score_sde_pytorch.configs.ve import cifar10_ncsnpp_continuous as configs
from score_based_model.score_sde_pytorch.sde_lib import VESDE


def get_cifar10_score_model(ckpt_path, device="cuda"):
    config = configs.get_config()
    sde = VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=config.model.num_scales,
    )
    score_model = mutils.create_model(config)

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_path, state, device)
    ema.copy_to(score_model.parameters())

    score_model = mutils.get_score_fn(sde, score_model, train=False, continuous=config.training.continuous)

    print("Loaded image score model.")
    return sde, score_model
