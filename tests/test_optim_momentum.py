import copy
import math
import unittest

import torch

from src.optim import IDBD, IDBDMomentum
from src.optim.IDBDMomentum import IDBDMomentum as ModuleIDBDMomentum


class IDBDMomentumTests(unittest.TestCase):
    def test_public_import(self):
        self.assertIs(IDBDMomentum, ModuleIDBDMomentum)
        self.assertIsInstance(IDBDMomentum([torch.nn.Parameter(torch.zeros(2))]), IDBD)

    def test_updates_match_equations_for_vectors_batches_and_multiple_outputs(self):
        for batched in (False, True):
            for outputs in (1, 2):
                with self.subTest(batched=batched, outputs=outputs):
                    weight = torch.nn.Parameter(
                        torch.tensor([[0.5, -0.25]], dtype=torch.float64).repeat(outputs, 1)
                    )
                    optimizer = IDBDMomentum(
                        [weight], initial_lr=0.1, meta_lr=0.2, beta_momentum=0.8
                    )
                    expected = weight.detach().clone()
                    beta = torch.full_like(weight, math.log(0.1))
                    h = torch.zeros_like(weight)
                    velocity = torch.zeros_like(weight)
                    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=weight.dtype)
                    if not batched:
                        x = x[0]
                    curvature = x.square().mean(0) if batched else x.square()
                    for _ in range(5):
                        optimizer.zero_grad()
                        prediction = x @ weight.T
                        # Sum over outputs, mean over examples.
                        (0.5 * (prediction - 1).square().sum(-1).mean()).backward()
                        gradient = weight.grad.detach().clone()
                        velocity = 0.8 * velocity - gradient * h
                        beta = beta + 0.2 * velocity
                        alpha = beta.exp()
                        expected = expected - alpha * gradient
                        h = h * (1 - alpha * curvature).clamp_min(0) - alpha * gradient
                        optimizer.step(x)
                        torch.testing.assert_close(weight, expected)
                        for key, value in (("beta", beta), ("h", h),
                                           ("beta_momentum_buffer", velocity)):
                            torch.testing.assert_close(optimizer.state[weight][key], value)

    def test_zero_momentum_exactly_recovers_idbd(self):
        weight = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
        other = torch.nn.Parameter(weight.detach().clone())
        plain = IDBD([weight], initial_lr=0.1, meta_lr=0.2)
        momentum = IDBDMomentum([other], initial_lr=0.1, meta_lr=0.2, beta_momentum=0)
        for gradient in ([0.3, -0.1], [-0.2, 0.4], [0., 0.], [0.1, 0.2]):
            weight.grad = torch.tensor(gradient, dtype=weight.dtype)
            other.grad = weight.grad.clone()
            plain.step(torch.ones(2))
            momentum.step(torch.ones(2))
            torch.testing.assert_close(weight, other, rtol=0, atol=0)
            self.assertEqual(plain.state[weight].keys(), momentum.state[other].keys())
            for key in plain.state[weight]:
                torch.testing.assert_close(plain.state[weight][key],
                                           momentum.state[other][key], rtol=0, atol=0)

    def test_zero_meta_lr_recovers_sgd(self):
        weight = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
        other = torch.nn.Parameter(weight.detach().clone())
        momentum = IDBDMomentum([weight], initial_lr=0.1, meta_lr=0)
        sgd = torch.optim.SGD([other], lr=0.1)
        for gradient in ([0.3, -0.1], [-0.2, 0.4], [0.1, 0.2]):
            weight.grad = torch.tensor(gradient, dtype=weight.dtype)
            other.grad = weight.grad.clone()
            momentum.step(torch.ones(2))
            sgd.step()
            torch.testing.assert_close(weight, other)
            torch.testing.assert_close(momentum.state[weight]["beta"].exp(),
                                       torch.full_like(weight, 0.1))

    def test_zero_gradient_decays_buffer_but_missing_gradient_skips_parameter(self):
        weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
        optimizer = IDBDMomentum([weight], initial_lr=0.1, meta_lr=0.2, beta_momentum=0.8)
        for _ in range(2):
            weight.grad = torch.ones_like(weight)
            optimizer.step(torch.ones(1))
        old_weight = weight.detach().clone()
        old_state = copy.deepcopy(optimizer.state[weight])
        weight.grad = torch.zeros_like(weight)
        optimizer.step(torch.zeros(1))
        torch.testing.assert_close(weight, old_weight)
        torch.testing.assert_close(optimizer.state[weight]["beta_momentum_buffer"],
                                   0.8 * old_state["beta_momentum_buffer"])
        torch.testing.assert_close(optimizer.state[weight]["beta"],
                                   old_state["beta"] + 0.2 * 0.8 * old_state["beta_momentum_buffer"])
        old_state = copy.deepcopy(optimizer.state[weight])
        weight.grad = None
        optimizer.step(torch.ones(1))
        for key in old_state:
            torch.testing.assert_close(optimizer.state[weight][key], old_state[key])

    def test_state_dict_resume_preserves_trajectory_and_group_options(self):
        weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
        optimizer = IDBDMomentum([weight], initial_lr=0.1, meta_lr=0.2, beta_momentum=0.7)
        for _ in range(3):
            weight.grad = torch.ones_like(weight)
            optimizer.step(torch.ones(1))
        restored = torch.nn.Parameter(weight.detach().clone())
        resumed = IDBDMomentum([restored])
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertEqual(resumed.param_groups[0]["beta_momentum"], 0.7)
        for _ in range(3):
            for p, opt in ((weight, optimizer), (restored, resumed)):
                p.grad = torch.full_like(p, -0.2)
                opt.step(torch.ones(1))
            torch.testing.assert_close(weight, restored, rtol=0, atol=0)
            for key in optimizer.state[weight]:
                torch.testing.assert_close(optimizer.state[weight][key],
                                           resumed.state[restored][key], rtol=0, atol=0)

    def test_parameter_groups_have_independent_momentum_and_lazy_state(self):
        first = torch.nn.Parameter(torch.tensor([0.5]))
        second = torch.nn.Parameter(first.detach().clone())
        skipped = torch.nn.Parameter(first.detach().clone())
        optimizer = IDBDMomentum([
            {"params": [first], "beta_momentum": 0},
            {"params": [second, skipped], "beta_momentum": 0.8},
        ], initial_lr=0.1, meta_lr=0.2)
        for _ in range(3):
            first.grad = torch.ones_like(first)
            second.grad = torch.ones_like(second)
            optimizer.step(torch.ones(1))
        self.assertNotIn("beta_momentum_buffer", optimizer.state[first])
        self.assertIn("beta_momentum_buffer", optimizer.state[second])
        self.assertNotIn(skipped, optimizer.state)
        self.assertGreater(optimizer.state[second]["beta"].item(),
                           optimizer.state[first]["beta"].item())

    def test_closure_return_and_inherited_feature_validation(self):
        weight = torch.nn.Parameter(torch.tensor([0.5]))
        optimizer = IDBDMomentum([weight])

        def closure():
            optimizer.zero_grad()
            loss = 0.5 * weight.square().sum()
            loss.backward()
            return loss

        loss = optimizer.step(torch.ones(1), closure=closure)
        self.assertEqual(loss.item(), 0.125)
        self.assertLess(weight.item(), 0.5)
        for x in (torch.empty(0, 1), torch.ones(1, 1, 1)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                optimizer.step(x)

    def test_invalid_momentum_is_rejected(self):
        for value in (-0.1, 1., 2., float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                IDBDMomentum([torch.nn.Parameter(torch.zeros(1))], beta_momentum=value)
            with self.subTest(group_value=value), self.assertRaises(ValueError):
                IDBDMomentum([{"params": [torch.nn.Parameter(torch.zeros(1))],
                               "beta_momentum": value}])


if __name__ == "__main__":
    unittest.main()
