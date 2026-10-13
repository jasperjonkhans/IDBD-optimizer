"""Learning-rate sweep using main's training/evaluation APIs and nested MLflow runs.

Example: uv run python -m src.sweep --initial-lrs 1e-4 1e-3 --meta-lrs 0 1e-2
"""

import argparse
import csv
import math
import time
from itertools import product
from pathlib import Path

import mlflow
import torch

if __package__:
    from . import main as experiment
    from .data import DataStream
else:  # Also support `python src/sweep.py`.
    import main as experiment
    from data import DataStream


SUMMARY_FIELDS = [
    "initial_lr", "meta_lr", "seed", "run_id", "model", "status",
    "half_squared_loss", "noisy_mse", "clean_mse", "active_clean_mse",
    "test_examples", "active_examples", "train_clean_mse", "train_noisy_mse",
    "useful_weight", "distractor_weight_rms", "useful_lr", "distractor_lr_mean",
    "seconds", "error",
]


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def weight_metrics(model, optimizer):
    weights = model.weight.detach().flatten()
    rates = experiment.learning_rates(model, optimizer).flatten()
    return dict(
        useful_weight=weights[0].item(),
        distractor_weight_rms=weights[1:].square().mean().sqrt().item(),
        useful_lr=rates[0].item(),
        distractor_lr_mean=rates[1:].mean().item(),
    )


def run_config(args, initial_lr, meta_lr, report):
    """Run one comparison inside the caller's active MLflow run."""
    # Same initial weights and fresh identical streams for every configuration.
    torch.manual_seed(args.seed)
    models, optimizers = experiment.create_comparison(
        args.input_dim, initial_lr, meta_lr, zeros=args.zeros,
    )
    models, optimizers, _, train_metrics, _ = experiment.train(
        DataStream(input_dim=args.input_dim, seed=args.seed),
        iterations=args.iterations, batch_size=args.batch_size,
        models=models, optimizers=optimizers, log_every=args.log_every,
        record_history=False,
        progress_callback=lambda step, values: report(
            "train", step, args.iterations, values
        ),
    )
    final_weights = {name: weight_metrics(model, optimizers[name])
                     for name, model in models.items()}
    test_metrics = experiment.evaluate(
        models, DataStream(input_dim=args.input_dim, seed=args.seed + 1),
        iterations=args.test_iterations, batch_size=args.test_batch_size,
        progress_callback=lambda step, values: report(
            "test", step, args.test_iterations,
            {name: {**final_weights[name], **metrics} for name, metrics in values.items()},
        ),
    )
    experiment.log_summary(test_metrics, "test", args.iterations)
    for name, values in final_weights.items():
        mlflow.log_metrics({f"final.{name}.{key}": value for key, value in values.items()})
    for name, table in (("train_metrics", train_metrics), ("test_metrics", test_metrics)):
        mlflow.log_text(table.to_csv(), f"data/{name}.csv")

    rows = []
    for name, metrics in test_metrics.iterrows():
        weights = final_weights.get(name, dict(
            useful_weight=0.0, distractor_weight_rms=0.0,
            useful_lr="", distractor_lr_mean="",
        ))
        rows.append(dict(
            model=name, half_squared_loss=metrics["half_squared_loss"],
            noisy_mse=metrics["noisy_mse"], clean_mse=metrics["clean_mse"],
            active_clean_mse=(metrics["active_clean_mse"]
                              if math.isfinite(metrics["active_clean_mse"]) else ""),
            test_examples=int(metrics["examples"]),
            active_examples=int(metrics["active_examples"]),
            train_clean_mse=train_metrics.loc[name, "clean_mse"],
            train_noisy_mse=train_metrics.loc[name, "noisy_mse"], **weights,
        ))
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-lrs", nargs="+", type=positive_float,
                        default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--meta-lrs", nargs="+", type=nonnegative_float,
                        default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--iterations", "--steps", type=positive_int, default=2**18,
                        help="training updates per run (--steps is a legacy alias)")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--test-iterations", "--test-steps", type=positive_int, default=100)
    parser.add_argument("--test-batch-size", type=positive_int, default=128)
    parser.add_argument("--input-dim", type=positive_int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--zeros", action="store_true", help="use zero instead of random weights")
    parser.add_argument("--log-every", type=positive_int, default=100)
    parser.add_argument("--log-seconds", type=positive_float, default=5)
    parser.add_argument("--threads", type=positive_int, default=1, help="PyTorch CPU threads")
    parser.add_argument("--tracking-uri", default="http://100.96.85.78:5000")
    parser.add_argument("--experiment", default="SGD-vs-IDBD")
    parser.add_argument("--output", type=Path, default=Path("sweep_results.csv"))
    args = parser.parse_args(argv)
    if args.input_dim < 2:
        parser.error("--input-dim must be at least 2 (useful feature + distractors)")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    configs = list(product(args.initial_lrs, args.meta_lrs))
    per_run = args.iterations + args.test_iterations
    planned = len(configs) * per_run
    completed = successful = diverged = 0
    start = time.perf_counter()
    print(f"{len(configs)} Läufe, je {args.iterations} Trainingsupdates (Batch {args.batch_size}), "
          f"{args.test_iterations} Testbatches. CPU, {args.threads} Thread(s).", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name="Learning-rate sweep") as parent:
        mlflow.log_params({**vars(args), "configurations": len(configs)})
        try:
            with args.output.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                output.flush()
                for run, (initial_lr, meta_lr) in enumerate(configs, 1):
                    run_start = time.perf_counter()
                    last_log = run_start
                    run_done = 0
                    print(f"\nLauf {run}/{len(configs)}: initial_lr={initial_lr:g}, "
                          f"meta_lr={meta_lr:g}", flush=True)

                    def report(phase, step, steps, values):
                        nonlocal completed, run_done, last_log
                        completed += 1
                        run_done += 1
                        now = time.perf_counter()
                        if step not in (1, steps) and now - last_log < args.log_seconds:
                            return
                        elapsed = now - start
                        run_elapsed = now - run_start
                        print(f"[{run}/{len(configs)}] {phase} {step}/{steps} ({step / steps:.1%}) | "
                              f"gesamt {completed / planned:.1%} | vergangen {duration(elapsed)} | "
                              f"Rest Lauf ~{duration(run_elapsed / run_done * (per_run - run_done))} | "
                              f"Rest gesamt ~{duration(elapsed / completed * (planned - completed))}",
                              flush=True)
                        for name, metrics in values.items():
                            print(f"{name}: loss={metrics['loss']:.5g}, "
                                  f"w[0]={metrics['useful_weight']:.5g}, "
                                  f"distractor_rms={metrics['distractor_weight_rms']:.5g}, "
                                  f"lr[0]={metrics['useful_lr']:.5g}, "
                                  f"distractor_lr={metrics['distractor_lr_mean']:.5g}", flush=True)
                        last_log = time.perf_counter()

                    child_id = ""
                    try:
                        with mlflow.start_run(
                            run_name=f"initial_lr={initial_lr:g}, meta_lr={meta_lr:g}", nested=True
                        ) as child:
                            child_id = child.info.run_id
                            mlflow.log_params(dict(
                                initial_lr=initial_lr, meta_lr=meta_lr, seed=args.seed,
                                test_seed=args.seed + 1, input_dim=args.input_dim,
                                iterations=args.iterations, batch_size=args.batch_size,
                                test_iterations=args.test_iterations,
                                test_batch_size=args.test_batch_size, log_every=args.log_every,
                                loss="mean half-squared error", bias=False,
                                initialization="zeros" if args.zeros else "shared random weights",
                                feature_probability=0.01, impulse_probability=0.01, noise_variance=5,
                            ))
                            try:
                                rows = run_config(args, initial_lr, meta_lr, report)
                            except FloatingPointError as exc:
                                mlflow.set_tags({"status": "diverged", "error": str(exc)})
                                raise  # MLflow marks the child FAILED, not FINISHED.
                            mlflow.set_tag("status", "ok")
                            mlflow.log_metric("seconds", time.perf_counter() - run_start)
                        status, error = "ok", ""
                        successful += 1
                    except FloatingPointError as exc:
                        status, error = "diverged", str(exc)
                        rows = [dict(model=name) for name in ("IDBD", "SGD", "Zero")]
                        planned -= per_run - run_done
                        diverged += 1
                        print(f"Lauf abgebrochen: {error}; weiter mit nächster Kombination.", flush=True)
                    seconds = time.perf_counter() - run_start
                    for row in rows:
                        writer.writerow(dict(
                            initial_lr=initial_lr, meta_lr=meta_lr, seed=args.seed,
                            run_id=child_id, status=status, seconds=seconds, error=error, **row,
                        ))
                        if status == "ok":
                            active_mse = row["active_clean_mse"]
                            active_text = f"{active_mse:.6g}" if active_mse != "" else "n/a"
                            print(f"Test {row['model']}: clean_mse={row['clean_mse']:.6g}, "
                                  f"noisy_mse={row['noisy_mse']:.6g}, "
                                  f"active_clean_mse={active_text}", flush=True)
                    output.flush()
        finally:
            # Also preserve completed configurations if interrupted mid-sweep.
            mlflow.log_artifact(str(args.output), artifact_path="data")
            mlflow.log_metrics(dict(successful_runs=successful, diverged_runs=diverged,
                                    seconds=time.perf_counter() - start))
    print(f"\nFertig nach {duration(time.perf_counter() - start)}. Ergebnisse: {args.output}", flush=True)
    print(f"MLflow sweep run: {parent.info.run_id}", flush=True)
    return parent.info.run_id


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen; bereits abgeschlossene Läufe stehen in der CSV.", flush=True)
        raise SystemExit(130)
