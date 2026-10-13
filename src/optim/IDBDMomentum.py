"""IDBD with heavy-ball momentum on log learning rates, not on weights."""

import math

import torch
from torch import Tensor
from torch.optim.optimizer import ParamsT

from .IDBD import IDBD, idbd


class IDBDMomentum(IDBD):
    """Accumulate the beta descent direction while retaining plain IDBD weights.

    For each weight, with the current gradient ``g`` and previous trace ``h``::

        v = beta_momentum * v - g * h
        beta = beta + meta_lr * v
        alpha = exp(beta)
        weight = weight - alpha * g
        h = h * max(0, 1 - alpha * E[x**2]) - alpha * g

    Backpropagate mean half-squared error and call ``step(x)`` exactly as for
    IDBD. The per-weight ``beta_momentum_buffer`` starts at zero and is included
    in ``state_dict()``. It decays on zero-gradient updates, but parameters with
    no gradient are skipped entirely. Momentum is unnormalized: a constant
    beta descent direction is amplified by ``1 / (1 - beta_momentum)`` in the
    long run. ``beta_momentum=0`` recovers IDBD; ``meta_lr=0`` recovers SGD.

    This is a heuristic meta-update, not an exact differentiation through the
    momentum dynamics. The original IDBD sensitivity-trace rule is unchanged.
    """

    def __init__(
        self,
        params: ParamsT,
        meta_lr: float | Tensor = 1e-3,
        initial_lr: float | Tensor = 1e-3,
        *,
        beta_momentum: float = 0.9,
    ) -> None:
        self._validate_momentum(beta_momentum)
        super().__init__(params, meta_lr=meta_lr, initial_lr=initial_lr)
        self.defaults["beta_momentum"] = float(beta_momentum)
        for group in self.param_groups:
            group.setdefault("beta_momentum", float(beta_momentum))
            self._validate_momentum(group["beta_momentum"])

    @staticmethod
    def _validate_momentum(value):
        if not math.isfinite(float(value)) or not 0 <= float(value) < 1:
            raise ValueError("beta_momentum must be finite and in [0, 1)")

    def _update_group(self, group, params, grads, betas, hs, x):
        momentum = float(group["beta_momentum"])
        self._validate_momentum(momentum)
        buffers = None
        if momentum != 0:
            buffers = []
            for p in params:
                state = self.state[p]
                if "beta_momentum_buffer" not in state:
                    state["beta_momentum_buffer"] = torch.zeros_like(p)
                buffers.append(state["beta_momentum_buffer"])
        idbd(
            params, grads, betas, hs, x, float(group["meta_lr"]),
            beta_momentum=momentum, beta_momentum_buffers=buffers,
        )
