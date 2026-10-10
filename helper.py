import torch
import numpy as np
from torch.utils.data import IterableDataset

class DataStream(IterableDataset):
    def __init__(self, input_dim=4096, sample_fn=None, seed=1234):
        super().__init__()
        self.input_dim = input_dim
        self.rng = np.random.default_rng(seed)
        self.sample_fn = (
            self._sample_fn if sample_fn is None else sample_fn
        )

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
        while True:
            yield self.sample_fn(self.rng, self.input_dim)