"""Reproducible synthetic data for the SGD/IDBD experiments."""

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class DataStream(IterableDataset):
    """Infinite stream of sparse features, noisy targets and clean targets.

    Only the first feature determines the clean target. The default sampler
    adds Gaussian noise (variance 5) and rare +/-1 impulses. A custom sampler
    receives ``(numpy_generator, input_dim)`` and returns ``(x, noisy_y, clean_y)``.
    Fresh instances with the same seed reproduce the same stream; iterating
    an existing instance continues it. DataLoader workers get independent RNGs.
    """

    def __init__(self, input_dim=4096, sample_fn=None, seed=1234):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.input_dim = input_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._worker_id = None
        self.sample_fn = self._sample_fn if sample_fn is None else sample_fn

    @staticmethod
    def _sample_fn(rng, input_dim):
        x = rng.binomial(1, 0.01, size=input_dim).astype(np.float32)
        clean_y = float(x[0])

        impulse = (
            rng.choice([-1, 1])
            if rng.random() < 0.01
            else 0
        )
        noise = rng.normal(0, np.sqrt(5))
        noisy_y = clean_y + impulse + noise

        return (
            torch.from_numpy(x),
            torch.tensor(noisy_y, dtype=torch.float32),
            torch.tensor(clean_y, dtype=torch.float32),
        )

    def __iter__(self):
        worker = get_worker_info()
        if worker is not None and self._worker_id != worker.id:
            self.rng = np.random.default_rng(
                np.random.SeedSequence(self.seed, spawn_key=(worker.id,))
            )
            self._worker_id = worker.id
        while True:
            yield self.sample_fn(self.rng, self.input_dim)