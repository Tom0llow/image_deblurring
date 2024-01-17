import torch
import math

from app.utils import save_estimateds


def get_score(x, t, score_fn, num_scales, batch_size):
    def get_batch_score(x_batch, t, score_fn, batch_size):
        vec_t = torch.ones(batch_size, device=t.device) * t
        vec_t = vec_t.to(x_batch.device)
        return score_fn(x_batch, vec_t)

    score = torch.empty(1, *x.size()[1:], device=x.device)

    N = math.ceil(num_scales // batch_size) - 1
    if N == 0:
        score = get_batch_score(x, t, score_fn, batch_size)
        return score

    idx = list(range(0, num_scales, batch_size))
    for i in range(N):
        x_batch = x[idx[i] : idx[i + 1]]
        batch_score = get_batch_score(x_batch, t, score_fn, batch_size)
        score = torch.cat((score, batch_score), 0)
    score = torch.cat((score, x[idx[i + 1] :]), 0)

    return score[1:]


def clip_grad_norm_(grad, max_norm, norm_type):
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == torch.inf:
        norms = [torch.linalg.vector_norm(g.detach(), torch.inf).to(grad.device) for g in grad]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grad])
        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(grad.device) for norm in norms]), norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    clip_coef_clamped_device = clip_coef_clamped.to(grad.device)
    for g in grad:
        g.detach().mul_(clip_coef_clamped_device)

    return total_norm


class EarlyStopping:
    def __init__(self, fname, path_to_save, patience=100, verbose=False):
        self.fname = fname
        self.path_to_save = path_to_save
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = float("inf")

    def __call__(self, loss, estimated_i, estimated_k=None):
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save(loss, estimated_i, estimated_k)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save(loss, estimated_i, estimated_k)
            self.counter = 0

    def save(self, loss, estimated_i, estimated_k=None):
        if self.verbose:
            print(f"Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving estimateds ...")
        save_estimateds(self.fname, self.path_to_save, estimated_i, estimated_k)
        self.loss_min = loss
