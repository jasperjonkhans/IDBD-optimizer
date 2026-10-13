"""IDBD with Adam on log learning rates, not on weights."""

import math

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT

from .IDBD import IDBD, _feature_curvature, _update_weight_and_trace


class IDBDAdam(IDBD):
    """Apply bias-corrected Adam to the beta meta-gradient ``q = g * h``.

    For each weight, using the previous sensitivity trace ``h``::

        m = beta1 * m + (1 - beta1) * q
        v = beta2 * v + (1 - beta2) * q**2
        beta = beta - meta_lr * (m / (1 - beta1**t)) / (
            sqrt(v / (1 - beta2**t)) + eps)
        alpha = exp(beta)
        weight = weight - alpha * g
        h = h * max(0, 1 - alpha * E[x**2]) - alpha * g

    ``betas``, ``eps`` and optional ``amsgrad`` have PyTorch Adam semantics,
    but operate only on log learning rates. AMSGrad uses the running maximum
    of the uncorrected second moment in the denominator. State is lazy and
    per weight, including a per-parameter step counter for bias correction;
    parameters without gradients are skipped, while zero gradients still
    decay the moments. All state is included in ``state_dict()``.

    Backpropagate mean half-squared error and call ``step(x)`` as for IDBD.
    ``meta_lr=0`` recovers SGD. Like IDBDMomentum, this is a heuristic
    meta-update: the original IDBD trace is retained, not differentiated
    through Adam's dynamics. Adam's ``betas`` are moment decay factors,
    distinct from IDBD's per-weight log learning rate ``state[p]["beta"]``.
    """

    def __init__(
        self,
        params: ParamsT,
        meta_lr: float | Tensor = 1e-3,
        initial_lr: float | Tensor = 1e-3,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        amsgrad: bool = False,
    ) -> None:
        options = {"betas": betas, "eps": eps, "amsgrad": amsgrad}
        self._validate_options(options)
        super().__init__(params, meta_lr=meta_lr, initial_lr=initial_lr)
        self.defaults.update(options)
        for group in self.param_groups:
            for key, value in options.items():
                group.setdefault(key, value)
            self._validate_options(group)

    @staticmethod
    def _validate_options(group):
        betas = group["betas"]
        if len(betas) != 2 or any(
            not math.isfinite(float(b)) or not 0 <= float(b) < 1 for b in betas
        ):
            raise ValueError("betas must contain two finite values in [0, 1)")
        if not math.isfinite(float(group["eps"])) or float(group["eps"]) < 0:
            raise ValueError("eps must be finite and nonnegative")

    def _update_group(self, group, params, grads, betas, hs, x):
        self._validate_options(group)
        curvature = _feature_curvature(x)
        beta1, beta2 = map(float, group["betas"])
        eps = float(group["eps"])
        meta_lr = float(group["meta_lr"])
        for param, grad, beta, h in zip(params, grads, betas, hs):
            state = self.state[param]
            if "beta_exp_avg" not in state:
                state["beta_step"] = 0
                state["beta_exp_avg"] = torch.zeros_like(param)
                state["beta_exp_avg_sq"] = torch.zeros_like(param)
            if group["amsgrad"] and "beta_max_exp_avg_sq" not in state:
                state["beta_max_exp_avg_sq"] = torch.zeros_like(param)

            state["beta_step"] += 1
            step = state["beta_step"]
            meta_grad = grad * h
            mean = state["beta_exp_avg"]
            square_mean = state["beta_exp_avg_sq"]
            mean.lerp_(meta_grad, 1 - beta1)
            square_mean.mul_(beta2).addcmul_(meta_grad, meta_grad, value=1 - beta2)
            variance = square_mean
            if group["amsgrad"]:
                variance = state["beta_max_exp_avg_sq"]
                torch.maximum(variance, square_mean, out=variance)
            denominator = (variance / (1 - beta2**step)).sqrt().add_(eps)
            beta.addcdiv_(mean, denominator, value=-meta_lr / (1 - beta1**step))
            _update_weight_and_trace(param, grad, beta, h, curvature)
