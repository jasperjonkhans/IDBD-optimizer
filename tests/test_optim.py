import math
import unittest

import torch

from exercises.idbd_exercise import IDBD as ExerciseIDBD
from src.optim import IDBD
from src.optim.IDBD import IDBD as ModuleIDBD


class IDBDTests(unittest.TestCase):
    def test_canonical_and_legacy_imports_share_one_implementation(self):
        self.assertIs(IDBD, ModuleIDBD)
        self.assertIs(IDBD, ExerciseIDBD)
        self.assertEqual(IDBD.__module__, "src.optim.IDBD")

    def test_two_updates_match_idbd_equations_for_batches(self):
        weight = torch.nn.Parameter(torch.tensor([[0.5, -0.25]], dtype=torch.float64))
        optimizer = IDBD([weight], initial_lr=0.1, meta_lr=0.2)
        expected = weight.detach().clone()
        beta = torch.full_like(weight, math.log(0.1))
        h = torch.zeros_like(weight)
        x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
        y = torch.tensor([1., 0., 1.], dtype=torch.float64)
        for _ in range(2):
            optimizer.zero_grad()
            prediction = (x @ weight.T).squeeze(-1)
            (0.5 * (prediction - y).square().mean()).backward()
            gradient = weight.grad.detach().clone()
            beta -= 0.2 * gradient * h
            alpha = beta.exp()
            expected -= alpha * gradient
            h = h * (1 - alpha * x.square().mean(dim=0)).clamp_min(0) - alpha * gradient
            optimizer.step(x)
            torch.testing.assert_close(weight, expected)
            torch.testing.assert_close(optimizer.state[weight]["beta"], beta)
            torch.testing.assert_close(optimizer.state[weight]["h"], h)

    def test_invalid_feature_shapes_fail(self):
        for x in (torch.empty(0, 2), torch.ones(1, 1, 2)):
            weight = torch.nn.Parameter(torch.zeros(1, 2))
            weight.grad = torch.ones_like(weight)
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                IDBD([weight]).step(x)


if __name__ == "__main__":
    unittest.main()
