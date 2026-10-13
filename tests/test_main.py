import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from src import main
from src.data import DataStream
from src.optim import IDBD


class TrainingTests(unittest.TestCase):
    def test_training_matches_notebook(self):
        # Execute the notebook's actual train() without its long experiment.
        notebook = json.loads(
            (Path(main.__file__).parents[1] / "notebooks" / "sgd_idbd_exercise.ipynb").read_text()
        )
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if "def train(dataset," in "".join(cell["source"]))
        namespace = dict(torch=torch, nn=nn, pd=pd, copy=main.copy, IDBD=IDBD)
        exec(source.split("train_stream =")[0], namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(1234)
            expected_models, expected_history, expected_metrics, expected_predictions = (
                namespace["train"](DataStream(input_dim=4), 8, 32, 0.01, 0.1)
            )
            torch.manual_seed(1234)
            with patch.object(main.mlflow, "log_metrics"):
                models, _, history, metrics, predictions = main.train(
                    DataStream(input_dim=4), 8, 32, 0.01, 0.1, log_every=3
                )
        for name in models:
            torch.testing.assert_close(models[name].weight, expected_models[name].weight)
        pd.testing.assert_frame_equal(history[expected_history.columns], expected_history)
        pd.testing.assert_frame_equal(metrics, expected_metrics)
        pd.testing.assert_frame_equal(predictions, expected_predictions)

    def test_stopping_records_first_hit_and_waits_for_both_models(self):
        calls = []

        def stop_condition(step, name, metrics):
            calls.append((step, name, metrics.copy()))
            return step >= {"IDBD": 2, "SGD": 4}[name]

        with patch.object(main.mlflow, "log_metrics") as log, \
                contextlib.redirect_stdout(io.StringIO()):
            _, _, history, metrics, predictions = main.train(
                DataStream(input_dim=4), 10, 3, 0.01, 0.1,
                log_every=7, stop_condition=stop_condition,
            )
        self.assertEqual(metrics["steps_to_target"].to_dict(),
                         {"IDBD": 2, "SGD": 4, "Zero": None})
        self.assertEqual(metrics["examples"].tolist(), [12, 12, 12])
        self.assertEqual(len(predictions), 12)
        self.assertEqual(history["update"].tolist(), [1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual([(step, name) for step, name, _ in calls],
                         [(1, "IDBD"), (1, "SGD"), (2, "IDBD"), (2, "SGD"),
                          (3, "SGD"), (4, "SGD")])
        for step, name, values in calls:
            row = history[(history["update"] == step) & (history["model"] == name)].iloc[0]
            for key, value in values.items():
                self.assertEqual(row[key], value)
        self.assertEqual(sorted({call.kwargs["step"] for call in log.call_args_list}),
                         [1, 4])  # Early stopping must log the final update, too.

    def test_stopping_respects_iteration_budget(self):
        with patch.object(main.mlflow, "log_metrics"), \
                contextlib.redirect_stdout(io.StringIO()):
            _, _, history, metrics, _ = main.train(
                DataStream(input_dim=4), 5, 3, 0.01, 0.1,
                stop_condition=lambda step, name, values: name == "IDBD",
            )
        self.assertEqual(metrics["steps_to_target"].to_dict(),
                         {"IDBD": 1, "SGD": None, "Zero": None})
        self.assertEqual(len(history), 10)
        self.assertEqual(history["update"].max(), 5)

    def test_stopping_can_finish_on_first_update(self):
        with patch.object(main.mlflow, "log_metrics"), \
                contextlib.redirect_stdout(io.StringIO()):
            _, _, history, metrics, _ = main.train(
                DataStream(input_dim=4), 5, 3, 0.01, 0.1,
                stop_condition=lambda step, name, values: True,
            )
        self.assertEqual(metrics["steps_to_target"].to_dict(),
                         {"IDBD": 1, "SGD": 1, "Zero": None})
        self.assertEqual(len(history), 2)
        self.assertEqual(metrics["examples"].tolist(), [3, 3, 3])

    def test_zero_meta_rate_matches_sgd(self):
        with patch.object(main.mlflow, "log_metrics"), \
                contextlib.redirect_stdout(io.StringIO()):
            models, optimizers, _, _, _ = main.train(
                DataStream(input_dim=4), 20, 32, 0.01, 0.0
            )
        torch.testing.assert_close(models["IDBD"].weight, models["SGD"].weight)
        torch.testing.assert_close(
            main.learning_rates(models["IDBD"], optimizers["IDBD"]),
            torch.full((4,), 0.01),
        )

    def test_bounded_history_and_progress_callbacks_preserve_results(self):
        with patch.object(main.mlflow, "log_metrics"), \
                contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(1234)
            expected_models, _, history, expected_metrics, predictions = main.train(
                DataStream(input_dim=4), 5, 3, 0.01, 0.1
            )
            calls = []
            torch.manual_seed(1234)
            models, _, empty_history, metrics, empty_predictions = main.train(
                DataStream(input_dim=4), 5, 3, 0.01, 0.1,
                record_history=False,
                progress_callback=lambda step, values: calls.append((step, values)),
            )
        self.assertTrue(empty_history.empty)
        self.assertTrue(empty_predictions.empty)
        self.assertFalse(history.empty)
        self.assertFalse(predictions.empty)
        pd.testing.assert_frame_equal(metrics, expected_metrics)
        for name in models:
            torch.testing.assert_close(models[name].weight, expected_models[name].weight)
        self.assertEqual([step for step, _ in calls], [1, 2, 3, 4, 5])
        for step, values in calls:
            for name, row_values in values.items():
                row = history[(history["update"] == step) & (history["model"] == name)].iloc[0]
                for key, value in row_values.items():
                    self.assertEqual(value, row[key])
        evaluation_calls = []
        main.evaluate(models, DataStream(input_dim=4), 2, 3,
                      progress_callback=lambda step, values: evaluation_calls.append((step, values)))
        self.assertEqual([step for step, _ in evaluation_calls], [1, 2])
        self.assertEqual(set(evaluation_calls[0][1]), {"IDBD", "SGD"})

    def test_evaluation_is_frozen_and_metrics_are_correct(self):
        model = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(model.weight)
        model.eval()

        def sample(rng, input_dim):
            return torch.ones(input_dim), torch.tensor(3.0), torch.tensor(1.0)

        weights_before = model.weight.detach().clone()
        metrics = main.evaluate({"SGD": model}, DataStream(2, sample_fn=sample), 2, 3)
        self.assertFalse(model.training)
        torch.testing.assert_close(model.weight, weights_before)
        for name in ("SGD", "Zero"):
            self.assertEqual(metrics.loc[name, "examples"], 6)
            self.assertEqual(metrics.loc[name, "active_examples"], 6)
            self.assertEqual(metrics.loc[name, "noisy_mse"], 9)
            self.assertEqual(metrics.loc[name, "half_squared_loss"], 4.5)
            self.assertEqual(metrics.loc[name, "clean_mse"], 1)
            self.assertEqual(metrics.loc[name, "active_clean_mse"], 1)

    def test_undefined_active_mse_is_not_logged(self):
        totals = main.metric_totals()
        main.accumulate_metrics(totals, torch.zeros(2), torch.ones(2), torch.zeros(2))
        metrics = main.summarize_metrics({"Zero": totals})
        self.assertTrue(np.isnan(metrics.loc["Zero", "active_clean_mse"]))
        with patch.object(main.mlflow, "log_metrics") as log:
            main.log_summary(metrics, "train", 1)
        self.assertNotIn("train.Zero.active_clean_mse", log.call_args.args[0])
        self.assertEqual(log.call_args.args[0]["train.Zero.examples"], 2)

    def test_nonfinite_loss_fails(self):
        def sample(rng, input_dim):
            return torch.ones(input_dim), torch.tensor(float("nan")), torch.tensor(1.0)

        with self.assertRaisesRegex(FloatingPointError, "non-finite loss"):
            main.train(DataStream(2, sample_fn=sample), 1, 1, 0.01, 0.1)

    def test_invalid_arguments(self):
        for args in (["--input-dim", "1"], ["--iterations", "0"],
                     ["--log-every", "0"], ["--initial-lr", "nan"],
                     ["--meta-lr", "-1"], ["--seed", "-1"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main.parse_args(args)

    def test_mlflow_run_contains_metrics_parameters_and_plots(self):
        previous_cwd = Path.cwd()
        previous_uri = mlflow.get_tracking_uri()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    run_id = main.main([
                        "--input-dim", "4", "--iterations", "4", "--batch-size", "32",
                        "--test-iterations", "2", "--test-batch-size", "128",
                        "--log-every", "3",
                    ])
                client = MlflowClient()
                run = client.get_run(run_id)
                self.assertEqual(run.info.status, "FINISHED")
                self.assertEqual(run.data.params["initial_lr"], "0.01")
                self.assertEqual(run.data.params["meta_lr"], "0.1")
                self.assertEqual(run.data.params["test_seed"], "1235")
                for name in ("IDBD", "SGD", "Zero"):
                    self.assertEqual(run.data.metrics[f"train.{name}.examples"], 128)
                    self.assertEqual(run.data.metrics[f"test.{name}.examples"], 256)
                    self.assertAlmostEqual(run.data.metrics[f"test.{name}.noisy_mse"],
                                           2 * run.data.metrics[f"test.{name}.half_squared_loss"])
                for name in ("IDBD", "SGD"):
                    for key in ("loss", "loss_ma_200", "useful_weight", "distractor_weight_rms",
                                "useful_lr", "distractor_lr_mean", "lr_min", "lr_max"):
                        points = client.get_metric_history(run_id, f"train.{name}.{key}")
                        self.assertEqual([point.step for point in points], [1, 3, 4])
                for filename in ("training.png", "predictions.png", "test_metrics.png"):
                    path = client.download_artifacts(run_id, f"plots/{filename}")
                    with Image.open(path) as image:
                        self.assertGreater(image.width, 500)
                        image.verify()
                history = pd.read_csv(client.download_artifacts(run_id, "data/train_history.csv"))
                self.assertEqual(len(history), 8)  # Every step, even when logging every 3.
                predictions = pd.read_csv(
                    client.download_artifacts(run_id, "data/final_predictions.csv")
                )
                self.assertEqual(len(predictions), 128)
                parameters = pd.read_csv(
                    client.download_artifacts(run_id, "data/final_parameters.csv")
                )
                self.assertEqual(len(parameters), 4)
                np.testing.assert_allclose(parameters["SGD_lr"], 0.01)
                self.assertTrue(np.isfinite(parameters.to_numpy()).all())
                for filename in ("train_metrics.csv", "test_metrics.csv"):
                    table = pd.read_csv(client.download_artifacts(run_id, f"data/{filename}"))
                    self.assertEqual(table["model"].tolist(), ["IDBD", "SGD", "Zero"])
            finally:
                os.chdir(previous_cwd)
                mlflow.set_tracking_uri(previous_uri)


if __name__ == "__main__":
    unittest.main()
