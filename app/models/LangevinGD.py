import torch
import math

from models.model import E


# mean-field Langevin Dynamics
class LangevinGD(torch.optim.Optimizer):
    def __init__(self, params, lambda_, eta_, m):
        defaults = dict(lambda_=lambda_, eta_=eta_, m=m)
        super(LangevinGD, self).__init__(params, defaults)

    def step(self, score_fn):
        assert len(self.param_groups) == 1
        for group in self.param_groups:
            for p in group["params"]:
                lambda_ = group["lambda_"]
                lr = group["m"] * group["eta_"]
                noise = torch.randn_like(p.data)
                grad = p.grad.data

                grad_stepsize = lr * 0.5
                score_stepsize = lr * lambda_

                p_mean = (1 - score_stepsize) * p.data + score_stepsize * score_fn - grad_stepsize * grad
                p.data = p_mean + math.sqrt(2 * score_stepsize) * noise

                # The last step does not include any noise
                return E(p_mean)
