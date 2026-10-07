import math
import unittest

import torch

from idbd_exercise import IDBD


class IDBDBatchTests(unittest.TestCase):
    def test_batch_updates_match_mean_loss_curvature(self):
        weight = torch.nn.Parameter(torch.tensor([[0.5, -0.25]], dtype=torch.float64))
        optimizer = IDBD([weight], initial_lr=0.1, meta_lr=0.2)
        beta = torch.full_like(weight, math.log(0.1))
        h = torch.zeros_like(weight)
        expected = weight.detach().clone()
        x = torch.tensor([[1., 2.], [-1., 0.], [3., -2.]], dtype=torch.float64)
        target = torch.tensor([[1.], [0.], [-1.]], dtype=torch.float64)
        curvature = (x.T @ x).diagonal().reshape_as(weight) / len(x)

        for _ in range(3):
            optimizer.zero_grad()
            (0.5 * (x @ weight.T - target).square().mean()).backward()
            grad = x.T @ (x @ expected.T - target) / len(x)
            grad = grad.T
            beta = beta - 0.2 * grad * h
            alpha = beta.exp()
            expected = expected - alpha * grad
            h = h * (1 - alpha * curvature).clamp_min(0) - alpha * grad
            optimizer.step(x)
            torch.testing.assert_close(weight, expected)
            torch.testing.assert_close(optimizer.state[weight]["beta"], beta)
            torch.testing.assert_close(optimizer.state[weight]["h"], h)

    def test_single_example_and_size_one_batch_are_equivalent(self):
        weights = [torch.nn.Parameter(torch.tensor([[0.5, -0.25]])) for _ in range(2)]
        optimizers = [IDBD([w], initial_lr=0.1) for w in weights]
        vector = torch.tensor([1., 2.])
        for _ in range(3):
            for weight, optimizer, x in zip(weights, optimizers, (vector, vector[None])):
                optimizer.zero_grad()
                (0.5 * (x @ weight.T - 1).square().mean()).backward()
                optimizer.step(x)
            torch.testing.assert_close(weights[0], weights[1])
            for key in ("beta", "h"):
                torch.testing.assert_close(optimizers[0].state[weights[0]][key],
                                           optimizers[1].state[weights[1]][key])

    def test_zero_meta_rate_matches_batch_sgd(self):
        weights = [torch.nn.Parameter(torch.zeros(1, 3)) for _ in range(2)]
        optimizers = [IDBD([weights[0]], initial_lr=0.1, meta_lr=0),
                      torch.optim.SGD([weights[1]], lr=0.1)]
        x = torch.tensor([[1., 2., 0.], [-1., 0., 3.]])
        for _ in range(3):
            for weight, optimizer in zip(weights, optimizers):
                optimizer.zero_grad()
                (0.5 * (x @ weight.T - 1).square().mean()).backward()
                optimizer.step(x) if isinstance(optimizer, IDBD) else optimizer.step()
            torch.testing.assert_close(weights[0], weights[1])


if __name__ == "__main__":
    unittest.main()
