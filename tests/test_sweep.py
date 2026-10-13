import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlflow
from mlflow import MlflowClient
import pandas as pd
import torch

from src import main, sweep
from src.data import DataStream


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.previous_cwd = Path.cwd()
        self.previous_uri = mlflow.get_tracking_uri()
        self.previous_threads = torch.get_num_threads()
        self.directory = tempfile.TemporaryDirectory()
        os.chdir(self.directory.name)
        self.argv = [
            "--initial-lrs", "0.01", "0.02", "--meta-lrs", "0",
            "--input-dim", "4", "--iterations", "4", "--batch-size", "8",
            "--test-iterations", "2", "--test-batch-size", "64",
            "--log-every", "3", "--experiment", "sweep-test",
            "--output", "results/summary.csv",
            "--tracking-uri", f"sqlite:///{Path(self.directory.name) / 'mlflow.db'}",
        ]

    def tearDown(self):
        mlflow.end_run()
        mlflow.set_tracking_uri(self.previous_uri)
        torch.set_num_threads(self.previous_threads)
        os.chdir(self.previous_cwd)
        self.directory.cleanup()

    def test_tracking_uri_default_and_override(self):
        self.assertEqual(sweep.parse_args([]).tracking_uri, "http://100.96.85.78:5000")
        self.assertEqual(sweep.parse_args(self.argv).tracking_uri,
                         f"sqlite:///{Path(self.directory.name) / 'mlflow.db'}")

    def test_nested_runs_csv_and_shared_training_api(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(main, "train", wraps=main.train) as train, \
                patch.object(main, "evaluate", wraps=main.evaluate) as evaluate:
            parent_id = sweep.main(self.argv)
        self.assertEqual(train.call_count, 2)
        self.assertEqual(evaluate.call_count, 2)
        self.assertFalse(train.call_args.kwargs["record_history"])
        self.assertEqual(set(train.call_args.kwargs["models"]), {"IDBD", "SGD"})
        self.assertEqual(set(train.call_args.kwargs["optimizers"]), {"IDBD", "SGD"})
        self.assertNotIn("zeros", train.call_args.kwargs)
        self.assertIsNone(mlflow.active_run())
        client = MlflowClient()
        parent = client.get_run(parent_id)
        runs = client.search_runs([parent.info.experiment_id])
        self.assertEqual(len(runs), 3)
        children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == parent_id]
        self.assertEqual(len(children), 2)
        self.assertEqual(parent.info.status, "FINISHED")
        self.assertEqual(parent.data.metrics["successful_runs"], 2)
        csv = pd.read_csv("results/summary.csv")
        self.assertEqual(len(csv), 6)
        self.assertEqual(set(csv["model"]), {"IDBD", "SGD", "Zero"})
        self.assertEqual(set(csv["status"]), {"ok"})
        artifact = client.download_artifacts(parent_id, "data/summary.csv")
        pd.testing.assert_frame_equal(csv, pd.read_csv(artifact))
        for child in children:
            self.assertEqual(child.info.status, "FINISHED")
            self.assertEqual(child.data.params["initialization"], "shared random weights")
            self.assertEqual(child.data.params["test_seed"], "1235")
            rows = csv[csv["run_id"] == child.info.run_id]
            for _, row in rows.iterrows():
                name = row["model"]
                self.assertEqual(row["test_examples"], 128)
                self.assertEqual(child.data.metrics[f"train.{name}.examples"], 32)
                self.assertEqual(child.data.metrics[f"test.{name}.examples"], 128)
                for metric in ("clean_mse", "noisy_mse", "half_squared_loss"):
                    self.assertAlmostEqual(row[metric], child.data.metrics[f"test.{name}.{metric}"])
            lr = float(child.data.params["initial_lr"])
            # Compare against the shared API with the same initialization/data.
            with patch.object(main.mlflow, "log_metrics"), \
                    contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(1234)
                models, _, _, _, _ = main.train(
                    DataStream(4), 4, 8, *main.create_comparison(4, lr, 0)
                )
            expected = main.evaluate(models, DataStream(4, seed=1235), 2, 64)
            for _, row in rows.iterrows():
                self.assertAlmostEqual(row["clean_mse"], expected.loc[row["model"], "clean_mse"])
            idbd = rows[rows["model"] == "IDBD"].iloc[0]
            sgd = rows[rows["model"] == "SGD"].iloc[0]
            self.assertAlmostEqual(idbd["useful_weight"], sgd["useful_weight"], places=6)
            points = client.get_metric_history(child.info.run_id, "train.IDBD.loss")
            self.assertEqual([point.step for point in points], [1, 3, 4])
            table = client.download_artifacts(child.info.run_id, "data/test_metrics.csv")
            self.assertEqual(pd.read_csv(table)["model"].tolist(), ["IDBD", "SGD", "Zero"])

    def test_divergent_configuration_fails_child_and_continues_sweep(self):
        actual_run = sweep.run_config
        count = 0

        def diverge_first(*args):
            nonlocal count
            count += 1
            if count == 1:
                raise FloatingPointError("IDBD: non-finite loss at update 1")
            return actual_run(*args)

        with patch.object(sweep, "run_config", side_effect=diverge_first), \
                contextlib.redirect_stdout(io.StringIO()):
            parent_id = sweep.main(self.argv)
        client = MlflowClient()
        parent = client.get_run(parent_id)
        children = client.search_runs([parent.info.experiment_id],
                                     f"tags.`mlflow.parentRunId` = '{parent_id}'")
        self.assertEqual(sorted(child.info.status for child in children), ["FAILED", "FINISHED"])
        failed = next(child for child in children if child.info.status == "FAILED")
        self.assertEqual(failed.data.tags["status"], "diverged")
        self.assertIn("non-finite", failed.data.tags["error"])
        self.assertEqual(parent.data.metrics["diverged_runs"], 1)
        self.assertEqual(parent.data.metrics["successful_runs"], 1)
        rows = pd.read_csv("results/summary.csv")
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(rows[rows["status"] == "diverged"]), 3)
        self.assertTrue(rows.loc[rows["status"] == "diverged", "clean_mse"].isna().all())

    def test_interrupt_preserves_completed_csv_as_parent_artifact(self):
        actual_run = sweep.run_config
        count = 0

        def interrupt_second(*args):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt
            return actual_run(*args)

        with patch.object(sweep, "run_config", side_effect=interrupt_second), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            sweep.main(self.argv)
        client = MlflowClient()
        experiment = client.get_experiment_by_name("sweep-test")
        parents = client.search_runs([experiment.experiment_id], "tags.`mlflow.runName` = 'Learning-rate sweep'")
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        self.assertEqual(parent.info.status, "FAILED")
        artifact = client.download_artifacts(parent.info.run_id, "data/summary.csv")
        self.assertEqual(len(pd.read_csv(artifact)), 3)
        self.assertIsNone(mlflow.active_run())

    def test_cli_aliases_and_invalid_arguments(self):
        args = sweep.parse_args(["--steps", "3", "--test-steps", "2", "--meta-lrs", "0"])
        self.assertEqual(args.iterations, 3)
        self.assertEqual(args.test_iterations, 2)
        self.assertEqual(args.meta_lrs, [0])
        for argv in (["--input-dim", "1"], ["--seed", "-1"], ["--steps", "0"],
                     ["--initial-lrs", "nan"], ["--initial-lrs", "0"],
                     ["--meta-lrs", "-1"], ["--meta-lrs", "nan"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                sweep.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
