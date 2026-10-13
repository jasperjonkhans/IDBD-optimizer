"""IDBD with RMSprop on log learning rates, not on weights."""

import math

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT

from .IDBD import IDBD, _feature_curvature, _update_weight_and_trace


class IDBDRMSprop(IDBD):
    """Apply RMSprop to the beta meta-gradient ``q = g * h``.

    For each weight, using the previous sensitivity trace ``h``::

        v = alpha * v + (1 - alpha) * q**2
        beta = beta - meta_lr * q / (sqrt(v) + eps)
        learning_rate = exp(beta)
        weight = weight - learning_rate * g
        h = h * max(0, 1 - learning_rate * E[x**2]) - learning_rate * g

    ``alpha``, ``eps``, ``momentum`` and ``centered`` have PyTorch RMSprop
    semantics, but operate only on log learning rates. ``alpha`` is the
    square-average decay factor, not the weight learning rate. Centered
    RMSprop subtracts the squared moving mean of ``q`` before taking the
    square root. Optional momentum accumulates the normalized meta-gradient.

    Backpropagate mean half-squared error and call ``step(x)`` as for IDBD.
    State is lazy, per weight and included in ``state_dict()``. Parameters
    without gradients are skipped; zero gradients still decay the averages.
    ``meta_lr=0`` recovers SGD. Like IDBDMomentum, this is a heuristic
    meta-update: the original IDBD trace is retained, not differentiated
    through RMSprop's dynamics.
    """

    def __init__(
        self,
        params: ParamsT,
        meta_lr: float | Tensor = 1e-3,
        initial_lr: float | Tensor = 1e-3,
        *,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        options = {
            "alpha": alpha, "eps": eps, "momentum": momentum, "centered": centered,
        }
        self._validate_options(options)
        super().__init__(params, meta_lr=meta_lr, initial_lr=initial_lr)
        self.defaults.update(options)
        for group in self.param_groups:
            for key, value in options.items():
                group.setdefault(key, value)
            self._validate_options(group)

    @staticmethod
    def _validate_options(group):
        alpha = float(group["alpha"])
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0, 1]")
        for name in ("eps", "momentum"):
            value = float(group[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def _update_group(self, group, params, grads, betas, hs, x):
        self._validate_options(group)
        curvature = _feature_curvature(x)
        decay = float(group["alpha"])
        eps = float(group["eps"])
        momentum = float(group["momentum"])
        meta_lr = float(group["meta_lr"])
        for param, grad, beta, h in zip(params, grads, betas, hs):
            state = self.state[param]
            if "beta_square_avg" not in state:
                state["beta_square_avg"] = torch.zeros_like(param)
            if group["centered"] and "beta_grad_avg" not in state:
                state["beta_grad_avg"] = torch.zeros_like(param)
            if momentum > 0 and "beta_momentum_buffer" not in state:
                state["beta_momentum_buffer"] = torch.zeros_like(param)

            meta_grad = grad * h
            square_mean = state["beta_square_avg"]
            square_mean.mul_(decay).addcmul_(meta_grad, meta_grad, value=1 - decay)
            if group["centered"]:
                mean = state["beta_grad_avg"]
                mean.lerp_(meta_grad, 1 - decay)
                denominator = square_mean.addcmul(mean, mean, value=-1).sqrt_()
            else:
                denominator = square_mean.sqrt()
            denominator.add_(eps)
            if momentum > 0:
                buffer = state["beta_momentum_buffer"]
                buffer.mul_(momentum).addcdiv_(meta_grad, denominator)
                beta.add_(buffer, alpha=-meta_lr)
            else:
                beta.addcdiv_(meta_grad, denominator, value=-meta_lr)
            _update_weight_and_trace(param, grad, beta, h, curvature)
