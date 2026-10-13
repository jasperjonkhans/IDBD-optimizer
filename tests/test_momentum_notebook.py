"""Execute the comparison notebook headlessly with a small experiment budget."""

import contextlib
import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import torch


class MomentumNotebookTests(unittest.TestCase):
    def test_all_cells_run_and_comparisons_are_reproducible(self):
        root = Path(__file__).resolve().parents[1]
        notebook = nbformat.read(root / "notebooks/idbd_momentum.ipynb", as_version=4)
        nbformat.validate(notebook)
        namespace = {"__name__": "__notebook_test__"}
        rendered = []

        def render_and_close():
            for figure_number in plt.get_fignums():
                figure = plt.figure(figure_number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png")
                rendered.append(buffer.tell())
                plt.close(figure)

        try:
            with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, "show", render_and_close):
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type != "code":
                        continue
                    source = "\n".join(line for line in cell.source.splitlines()
                                       if not line.startswith("%"))
                    exec(compile(source, f"idbd_momentum.ipynb:cell{index}", "exec"), namespace)
                    if "RUN_MATCHED_META_LR = False" in cell.source:
                        namespace.update(INPUT_DIM=8, ITERATIONS=80, BATCH_SIZE=16,
                                         RECORD_EVERY=13, TEST_ITERATIONS=5,
                                         TEST_BATCH_SIZE=32, RUN_MATCHED_META_LR=True)

                result = namespace["results"]
                matched = namespace["matched_results"]
                self.assertEqual(len(rendered), 5)
                self.assertTrue(all(size > 1000 for size in rendered))
                self.assertEqual(list(result["trace"]), ["SGD", "IDBD", "IDBD + β-momentum"])
                np.testing.assert_array_equal(result["steps"], [0, 13, 26, 39, 52, 65, 78, 80])
                np.testing.assert_array_equal(result["examples"], result["steps"] * 16)
                for name, trace in result["trace"].items():
                    self.assertEqual(trace["weights"].shape, (8, 8))
                    self.assertEqual(trace["rates"].shape, (8, 8))
                    self.assertTrue(np.isfinite(trace["weights"]).all())
                    self.assertTrue((trace["rates"] > 0).all())
                    np.testing.assert_array_equal(trace["weights"][0],
                                                  result["trace"]["SGD"]["weights"][0])
                    np.testing.assert_allclose(trace["rates"][0], 0.01)
                np.testing.assert_allclose(result["trace"]["SGD"]["rates"], 0.01)
                for name in ("SGD", "IDBD"):
                    for key in result["trace"][name]:
                        np.testing.assert_array_equal(result["trace"][name][key],
                                                      matched["trace"][name][key])
                momentum_group = matched["optimizers"]["IDBD + β-momentum"].param_groups[0]
                self.assertAlmostEqual(momentum_group["meta_lr"], 0.01)

                rng = torch.get_rng_state().clone()
                repeated = namespace["compare"](**namespace["config"], momentum_meta_lr=0.1)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for name, trace in result["trace"].items():
                    for key in trace:
                        np.testing.assert_array_equal(trace[key], repeated["trace"][name][key])
                np.testing.assert_array_equal(result["metrics"], repeated["metrics"])

                # Evaluation must not change parameters, rates or momentum buffers.
                weights_before = {name: model.weight.detach().clone()
                                  for name, model in result["models"].items()}
                state_before = {name: copy.deepcopy(opt.state_dict())
                                for name, opt in result["optimizers"].items()}
                namespace["evaluate"](result["models"],
                                      namespace["DataStream"](input_dim=8, seed=99), 3, 16)
                for name, model in result["models"].items():
                    torch.testing.assert_close(model.weight, weights_before[name], rtol=0, atol=0)
                    after = result["optimizers"][name].state_dict()
                    self.assertEqual(after["param_groups"], state_before[name]["param_groups"])
                    for parameter, state in state_before[name]["state"].items():
                        for key, value in state.items():
                            torch.testing.assert_close(after["state"][parameter][key], value,
                                                       rtol=0, atol=0)
        finally:
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
