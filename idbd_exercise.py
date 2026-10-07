import math

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class IDBD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        meta_lr: float | Tensor = 1e-3,
        initial_lr: float | Tensor = 1e-3,
    ) -> None:
        defaults = {
            "meta_lr": meta_lr,
            "initial_lr": initial_lr,
        }
        super().__init__(params, defaults)

    def _init_group(self, group, params, grads, betas, hs):
        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if "beta" not in state:
                state["beta"] = torch.full_like(p, math.log(group["initial_lr"]))
                state["h"] = torch.zeros_like(p)

            params.append(p)
            grads.append(p.grad)
            betas.append(state["beta"])
            hs.append(state["h"])

    @torch.no_grad()
    def step(self, x, closure=None):
        """Update a linear predictor from a feature vector or a batch of vectors.

        For batches, gradients must come from mean half-squared error over
        examples (sum over outputs for a multi-output predictor).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            betas: list[Tensor] = []
            hs: list[Tensor] = []

            self._init_group(group, params, grads, betas, hs)
            idbd(params, grads, betas, hs, x, float(group["meta_lr"]))
        return loss


def idbd(
    params: list[Tensor],
    grads: list[Tensor],
    betas: list[Tensor],
    hs: list[Tensor],
    x: Tensor,
    meta_lr: float,
) -> None:
    _single_tensor_idbd(params, grads, betas, hs, x, meta_lr)


def _single_tensor_idbd(
    params: list[Tensor],
    grads: list[Tensor],
    betas: list[Tensor],
    hs: list[Tensor],
    x: Tensor,
    meta_lr: float,
) -> None:
    if x.ndim not in (1, 2):
        raise ValueError("x must have shape (features,) or (batch, features)")
    if x.numel() == 0:
        raise ValueError("x must contain at least one example and feature")

    # The diagonal curvature of mean half-squared error is E[x_i ** 2],
    # not E[x_i] ** 2. Keep optimizer state per weight, not per example.
    feature_curvature = x.square()
    if x.ndim == 2:
        feature_curvature = feature_curvature.mean(dim=0)

    for param, grad, beta, h in zip(params, grads, betas, hs):
        curvature = feature_curvature.to(device=param.device, dtype=param.dtype)
        if param.numel() == curvature.numel():
            curvature = curvature.reshape_as(param)
        else:
            curvature = curvature.expand_as(param)
        beta.addcmul_(grad, h, value=-meta_lr)
        alpha = beta.exp()
        update = alpha * grad
        param.sub_(update)
        h.mul_((1 - alpha * curvature).clamp_min_(0)).sub_(update)
