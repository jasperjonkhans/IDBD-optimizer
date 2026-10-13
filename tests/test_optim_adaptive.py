import copy
import math
import unittest

import torch

from src.optim import IDBD, IDBDAdam, IDBDRMSprop
from src.optim.IDBDAdam import IDBDAdam as ModuleIDBDAdam
from src.optim.IDBDRMSprop import IDBDRMSprop as ModuleIDBDRMSprop


# Exercise standard algorithms and their optional variants against PyTorch.
CONFIGURATIONS = [
    (IDBDAdam, torch.optim.Adam, {}),
    (IDBDAdam, torch.optim.Adam, {"betas": (0.6, 0.8), "eps": 1e-5}),
    (IDBDAdam, torch.optim.Adam, {"betas": (0.6, 0.8), "amsgrad": True}),
    (IDBDRMSprop, torch.optim.RMSprop, {}),
    *[(IDBDRMSprop, torch.optim.RMSprop,
       {"alpha": 0.8, "eps": 1e-5, "momentum": momentum, "centered": centered})
      for momentum in (0., 0.7) for centered in (False, True)],
]


class IDBDAdaptiveTests(unittest.TestCase):
    def test_public_imports_and_inheritance(self):
        self.assertIs(IDBDAdam, ModuleIDBDAdam)
        self.assertIs(IDBDRMSprop, ModuleIDBDRMSprop)
        for cls in (IDBDAdam, IDBDRMSprop):
            self.assertIsInstance(cls([torch.nn.Parameter(torch.zeros(2))]), IDBD)

    def test_meta_updates_match_pytorch_and_weights_keep_idbd_equations(self):
        for cls, reference_cls, options in CONFIGURATIONS:
            for shape in ((2,), (1, 2), (2, 2)):
                for batched in (False, True):
                    with self.subTest(cls=cls, options=options, shape=shape, batched=batched):
                        weight = torch.nn.Parameter(torch.full(shape, 0.5, dtype=torch.float64))
                        optimizer = cls([weight], initial_lr=0.1, meta_lr=0.02, **options)
                        beta = torch.nn.Parameter(torch.full_like(weight, math.log(0.1)))
                        reference = reference_cls([beta], lr=0.02, **options)
                        expected_weight = weight.detach().clone()
                        expected_h = torch.zeros_like(weight)
                        x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=weight.dtype)
                        if not batched:
                            x = x[0]
                        curvature = x.square().mean(0) if batched else x.square()

                        # Missing gradients skip the parameter; explicit zeros do not.
                        for values in ([0.3, -0.1], [-0.2, 0.4], None,
                                       [0., 0.], [0.1, 0.2], [-0.4, -0.2]):
                            if values is None:
                                weight.grad = None
                                beta.grad = None
                            else:
                                grad = torch.tensor(values, dtype=weight.dtype).expand_as(weight).clone()
                                weight.grad = grad
                                beta.grad = grad * expected_h
                            reference.step()
                            if values is not None:
                                update = beta.detach().exp() * weight.grad
                                expected_weight -= update
                                expected_h = expected_h * (1 - beta.detach().exp() * curvature).clamp_min(0) - update
                            optimizer.step(x)
                            torch.testing.assert_close(weight, expected_weight)
                            torch.testing.assert_close(optimizer.state[weight]["beta"], beta)
                            torch.testing.assert_close(optimizer.state[weight]["h"], expected_h)
                            key_pairs = (
                                [("beta_step", "step"), ("beta_exp_avg", "exp_avg"),
                                 ("beta_exp_avg_sq", "exp_avg_sq"),
                                 ("beta_max_exp_avg_sq", "max_exp_avg_sq")]
                                if cls is IDBDAdam else
                                [("beta_square_avg", "square_avg"), ("beta_grad_avg", "grad_avg"),
                                 ("beta_momentum_buffer", "momentum_buffer")]
                            )
                            for ours, theirs in key_pairs:
                                if theirs in reference.state[beta]:
                                    actual = optimizer.state[weight][ours]
                                    expected = reference.state[beta][theirs]
                                    if ours == "beta_step":
                                        self.assertEqual(actual, expected.item())
                                    else:
                                        torch.testing.assert_close(actual, expected)

    def test_zero_meta_lr_recovers_sgd(self):
        for cls, _, options in CONFIGURATIONS:
            with self.subTest(cls=cls, options=options):
                weight = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
                other = torch.nn.Parameter(weight.detach().clone())
                optimizer = cls([weight], initial_lr=0.1, meta_lr=0, **options)
                sgd = torch.optim.SGD([other], lr=0.1)
                for values in ([0.3, -0.1], [-0.2, 0.4], [0., 0.], [0.1, 0.2]):
                    weight.grad = torch.tensor(values, dtype=weight.dtype)
                    other.grad = weight.grad.clone()
                    optimizer.step(torch.ones(2))
                    sgd.step()
                    torch.testing.assert_close(weight, other)
                    torch.testing.assert_close(optimizer.state[weight]["beta"].exp(),
                                               torch.full_like(weight, 0.1))

    def test_resume_preserves_all_state_and_group_options(self):
        for cls, _, options in CONFIGURATIONS:
            with self.subTest(cls=cls, options=options):
                weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
                optimizer = cls([weight], initial_lr=0.1, meta_lr=0.02, **options)
                for _ in range(3):
                    weight.grad = torch.ones_like(weight)
                    optimizer.step(torch.ones(1))
                restored = torch.nn.Parameter(weight.detach().clone())
                resumed = cls([restored])
                resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                for key, value in options.items():
                    self.assertEqual(resumed.param_groups[0][key], value)
                for _ in range(3):
                    for param, opt in ((weight, optimizer), (restored, resumed)):
                        param.grad = torch.full_like(param, -0.2)
                        opt.step(torch.ones(1))
                    torch.testing.assert_close(weight, restored, rtol=0, atol=0)
                    for key, value in optimizer.state[weight].items():
                        other = resumed.state[restored][key]
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, other, rtol=0, atol=0)
                        else:
                            self.assertEqual(value, other)

    def test_parameter_groups_and_late_gradients_have_independent_state(self):
        for cls, reference_cls, _ in (CONFIGURATIONS[0], CONFIGURATIONS[3]):
            with self.subTest(cls=cls):
                options = ({"betas": (0.5, 0.8), "amsgrad": True} if cls is IDBDAdam else
                           {"alpha": 0.8, "momentum": 0.7, "centered": True})
                first, second, skipped = [torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
                                          for _ in range(3)]
                optimizer = cls([
                    {"params": [first]},
                    {"params": [second, skipped], "meta_lr": 0.04, **options},
                ], initial_lr=0.1, meta_lr=0.02)
                ref_beta = torch.nn.Parameter(torch.full_like(second, math.log(0.1)))
                reference = reference_cls([ref_beta], lr=0.04, **options)
                for _ in range(3):
                    first.grad = torch.ones_like(first)
                    second.grad = torch.ones_like(second)
                    old_h = optimizer.state.get(second, {}).get("h", torch.zeros_like(second))
                    ref_beta.grad = second.grad * old_h
                    reference.step()
                    optimizer.step(torch.ones(1))
                    torch.testing.assert_close(optimizer.state[second]["beta"], ref_beta)
                self.assertNotIn(skipped, optimizer.state)
                skipped.grad = torch.ones_like(skipped)
                optimizer.step(torch.ones(1))
                torch.testing.assert_close(optimizer.state[skipped]["beta"],
                                           torch.full_like(skipped, math.log(0.1)))
                if cls is IDBDAdam:
                    self.assertEqual(optimizer.state[skipped]["beta_step"], 1)
                    self.assertEqual(optimizer.state[first]["beta_step"], 4)
                self.assertNotEqual(optimizer.state[first]["beta"].item(),
                                    optimizer.state[second]["beta"].item())
                # Defaults are also available to groups added after construction.
                late = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
                optimizer.add_param_group({"params": [late]})
                late.grad = torch.ones_like(late)
                optimizer.step(torch.ones(1))
                self.assertIn(late, optimizer.state)

    def test_closure_and_feature_validation(self):
        for cls in (IDBDAdam, IDBDRMSprop):
            with self.subTest(cls=cls):
                weight = torch.nn.Parameter(torch.tensor([0.5]))
                optimizer = cls([weight])

                def closure():
                    optimizer.zero_grad()
                    loss = 0.5 * weight.square().sum()
                    loss.backward()
                    return loss

                loss = optimizer.step(torch.ones(1), closure=closure)
                self.assertEqual(loss.item(), 0.125)
                self.assertLess(weight.item(), 0.5)
                snapshot = copy.deepcopy(optimizer.state[weight])
                for x in (torch.empty(0, 1), torch.ones(1, 1, 1)):
                    with self.assertRaises(ValueError):
                        optimizer.step(x)
                for key, value in snapshot.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(optimizer.state[weight][key], value, rtol=0, atol=0)
                    else:
                        self.assertEqual(optimizer.state[weight][key], value)

    def test_invalid_options_rejected_globally_and_per_group(self):
        bad_adam = [{"betas": values} for values in (
            (-0.1, 0.9), (0.9, 1.), (float("nan"), 0.9), (0.9,),
            (0.9, float("inf")),
        )]
        bad_rmsprop = [{"alpha": value} for value in (-0.1, 1.1, float("nan"), float("inf"))]
        bad_rmsprop += [{"momentum": value} for value in (-0.1, float("nan"), float("inf"))]
        for cls, invalid in ((IDBDAdam, bad_adam), (IDBDRMSprop, bad_rmsprop)):
            invalid += [{"eps": value} for value in (-1e-8, float("nan"), float("inf"))]
            for options in invalid:
                with self.subTest(cls=cls, options=options):
                    with self.assertRaises(ValueError):
                        cls([torch.nn.Parameter(torch.zeros(1))], **options)
                    with self.assertRaises(ValueError):
                        cls([{"params": [torch.nn.Parameter(torch.zeros(1))], **options}])


if __name__ == "__main__":
    unittest.main()
