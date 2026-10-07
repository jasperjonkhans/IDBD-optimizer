"""Exercise the notebook's actual training/test functions without long runs."""
import ast
import json
import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn


class NotebookTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(
            Path(__file__).with_name("sgd_idbd_exercise.ipynb").read_text()
        )
        cls.scope = dict(np=np, pd=pd, torch=torch, nn=nn)
        # Load definitions and imports, excluding training calls and plotting.
        for index in (6, 8, 9):
            tree = ast.parse("".join(notebook["cells"][index]["source"]))
            tree.body = [node for node in tree.body if isinstance(
                node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
            )]
            exec(compile(tree, f"notebook_cell_{index}", "exec"), cls.scope)

    def test_shared_training_stream_and_reproducibility(self):
        runs = [self.scope["train"](
            self.scope["DataStream"](input_dim=8, seed=56),
            iterations=100, batch_size=1, initial_lr=0.01, meta_lr=0,
        ) for _ in range(2)]
        for models, history, metrics, predictions in runs:
            torch.testing.assert_close(models["IDBD"].weight, models["SGD"].weight)
            np.testing.assert_allclose(predictions["IDBD"], predictions["SGD"],
                                       atol=1e-7)
            self.assertTrue(metrics["examples"].eq(100).all())
            self.assertEqual(len(history), 200)
            self.assertEqual(len(predictions), 100)
        torch.testing.assert_close(runs[0][0]["IDBD"].weight,
                                   runs[1][0]["IDBD"].weight)

    def test_evaluation_weights_partial_batches_and_preserves_model(self):
        class Examples(torch.utils.data.IterableDataset):
            def __iter__(self):
                for value in (0., 1., 0., 0., 1.):
                    yield (torch.tensor([value, 0.]), torch.tensor(value + 2),
                           torch.tensor(value))

        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.5, 0.]]))
        before = model.weight.detach().clone()
        metrics = self.scope["test"]({"fixture": model}, Examples(), 10, 3)
        self.assertEqual(metrics.loc["fixture", "examples"], 5)
        self.assertEqual(metrics.loc["fixture", "active_examples"], 2)
        self.assertAlmostEqual(metrics.loc["fixture", "clean_mse"], 0.1)
        self.assertAlmostEqual(metrics.loc["fixture", "active_clean_mse"], 0.25)
        self.assertAlmostEqual(metrics.loc["fixture", "noisy_mse"], 4.9)
        self.assertAlmostEqual(metrics.loc["Zero", "clean_mse"], 0.4)
        torch.testing.assert_close(model.weight, before)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)

    def test_no_active_examples_have_undefined_active_mse(self):
        class Examples(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield torch.zeros(2), torch.tensor(0.), torch.tensor(0.)

        model = nn.Linear(2, 1, bias=False)
        metrics = self.scope["test"]({"fixture": model}, Examples(), 1, 1)
        self.assertTrue(math.isnan(metrics.loc["fixture", "active_clean_mse"]))


if __name__ == "__main__":
    unittest.main()
