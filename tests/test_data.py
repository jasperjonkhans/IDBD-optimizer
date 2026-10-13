from itertools import islice
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import DataStream


def random_sample(rng, input_dim):
    return torch.from_numpy(rng.random(input_dim)), torch.tensor(0.), torch.tensor(0.)


class DataStreamTests(unittest.TestCase):
    def test_default_sampler_preserves_original_distribution_and_sequence(self):
        rng = np.random.default_rng(1234)
        samples = iter(DataStream(input_dim=16, seed=1234))
        for _ in range(20):
            x = rng.binomial(1, 0.01, size=16).astype(np.float32)
            impulse = rng.choice([-1, 1]) if rng.random() < 0.01 else 0
            noisy_y = float(x[0]) + impulse + rng.normal(0, np.sqrt(5))
            actual_x, actual_noisy, actual_clean = next(samples)
            torch.testing.assert_close(actual_x, torch.from_numpy(x))
            torch.testing.assert_close(actual_noisy, torch.tensor(noisy_y, dtype=torch.float32))
            self.assertEqual(actual_clean.item(), float(x[0]))

    def test_new_instances_reproduce_and_existing_stream_continues(self):
        stream = DataStream(4, sample_fn=random_sample)
        first = next(iter(stream))[0]
        second = next(iter(stream))[0]
        samples = list(islice(DataStream(4, sample_fn=random_sample), 2))
        torch.testing.assert_close(first, samples[0][0])
        torch.testing.assert_close(second, samples[1][0])
        self.assertFalse(torch.equal(first, second))

    def test_worker_streams_are_independent_reproducible_and_continue(self):
        worker_samples = []
        for worker_id in (0, 1):
            with patch("src.data.get_worker_info", return_value=SimpleNamespace(id=worker_id)):
                stream = DataStream(4, sample_fn=random_sample)
                first = next(iter(stream))[0]
                second = next(iter(stream))[0]
                expected = list(islice(DataStream(4, sample_fn=random_sample), 2))
                torch.testing.assert_close(first, expected[0][0])
                torch.testing.assert_close(second, expected[1][0])
                worker_samples.append(first)
        self.assertFalse(torch.equal(*worker_samples))

    def test_dataloader_workers_do_not_duplicate_samples(self):
        loader = DataLoader(DataStream(4, sample_fn=random_sample),
                            batch_size=None, num_workers=2)
        samples = list(islice(loader, 4))
        for worker_id in (0, 1):
            rng = np.random.default_rng(np.random.SeedSequence(1234, spawn_key=(worker_id,)))
            for index in (worker_id, worker_id + 2):
                torch.testing.assert_close(samples[index][0], random_sample(rng, 4)[0])
        self.assertFalse(torch.equal(samples[0][0], samples[1][0]))

    def test_invalid_arguments(self):
        for kwargs in ({"input_dim": 0}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DataStream(**kwargs)


if __name__ == "__main__":
    unittest.main()
