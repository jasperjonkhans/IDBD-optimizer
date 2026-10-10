"""Standalone learning-rate sweep; no notebook execution or imports.

Example: uv run python sweep.py --initial-lrs 1e-4 1e-3 --meta-lrs 1e-3 1e-2
"""

import argparse
import csv
import math
import time
from itertools import product
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from helper import DataStream
from idbd_exercise import IDBD


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


def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_config(args, initial_lr, meta_lr, report):
    models = {name: nn.Linear(args.input_dim, 1, bias=False)
              for name in ("IDBD", "SGD")}
    for model in models.values():
        nn.init.zeros_(model.weight)
    optimizers = {
        "IDBD": IDBD(models["IDBD"].parameters(), initial_lr=initial_lr,
                     meta_lr=meta_lr),
        "SGD": torch.optim.SGD(models["SGD"].parameters(), lr=initial_lr),
    }
    totals = {name: dict(noisy=0.0, clean=0.0, active=0.0)
              for name in (*models, "Zero")}
    examples = active_examples = 0
    for phase, steps, batch_size, seed in (
        ("train", args.steps, args.batch_size, args.seed),
        ("test", args.test_steps, args.test_batch_size, args.seed + 1),
    ):
        # Fresh streams give every configuration identical train/test data.
        loader = iter(DataLoader(DataStream(input_dim=args.input_dim, seed=seed),
                                 batch_size=batch_size, num_workers=0))
        training = phase == "train"
        for model in models.values():
            model.train(training)
        for step in range(1, steps + 1):
            x, noisy_y, clean_y = next(loader)
            losses = {}
            with torch.set_grad_enabled(training):
                predictions = {}
                for name, model in models.items():
                    optimizer = optimizers[name]
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    prediction = model(x).squeeze(-1)
                    loss = 0.5 * (prediction - noisy_y).square().mean()
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(f"{name}: non-finite loss ({phase}, {step})")
                    losses[name] = loss.item()
                    if training:
                        loss.backward()
                        if name == "IDBD":
                            optimizer.step(x=x)
                        else:
                            optimizer.step()
                        tensors = [model.weight, *optimizer.state[model.weight].values()]
                        if any(isinstance(t, torch.Tensor) and not torch.isfinite(t).all().item()
                               for t in tensors):
                            raise FloatingPointError(f"{name}: non-finite state (train, {step})")
                    predictions[name] = prediction.detach()
            if not training:
                examples += clean_y.numel()
                active = clean_y == 1
                active_examples += active.sum().item()
                predictions["Zero"] = torch.zeros_like(clean_y)
                for name, prediction in predictions.items():
                    clean_error = prediction.double() - clean_y.double()
                    totals[name]["noisy"] += (prediction.double() - noisy_y.double()).square().sum().item()
                    totals[name]["clean"] += clean_error.square().sum().item()
                    totals[name]["active"] += clean_error[active].square().sum().item()
            report(phase, step, steps, models, optimizers, losses)
    return [dict(model=name, noisy_mse=t["noisy"] / examples,
                 clean_mse=t["clean"] / examples,
                 active_clean_mse=t["active"] / active_examples if active_examples else "",
                 test_examples=examples, active_examples=active_examples)
            for name, t in totals.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-lrs", nargs="+", type=positive_float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--meta-lrs", nargs="+", type=positive_float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--steps", type=positive_int, default=2**18, help="training updates per run")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--test-steps", type=positive_int, default=100)
    parser.add_argument("--test-batch-size", type=positive_int, default=128)
    parser.add_argument("--input-dim", type=positive_int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-seconds", type=positive_float, default=5)
    parser.add_argument("--threads", type=positive_int, default=1, help="PyTorch CPU threads")
    parser.add_argument("--output", type=Path, default=Path("sweep_results.csv"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    configs = list(product(args.initial_lrs, args.meta_lrs))
    per_run = args.steps + args.test_steps
    planned = len(configs) * per_run
    completed = 0
    start = time.perf_counter()
    fields = ["initial_lr", "meta_lr", "seed", "model", "status", "noisy_mse",
              "clean_mse", "active_clean_mse", "test_examples", "active_examples", "seconds", "error"]
    print(f"{len(configs)} Läufe, je {args.steps} Trainingsupdates (Batch {args.batch_size}), "
          f"{args.test_steps} Testbatches. CPU, {args.threads} Thread(s).", flush=True)
    print("Restzeiten sind Schätzungen aus der bisherigen Geschwindigkeit; erste Schätzung nach Update 1.",
          flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run, (initial_lr, meta_lr) in enumerate(configs, 1):
            run_start = time.perf_counter()
            last_log = run_start
            run_done = 0
            print(f"\nLauf {run}/{len(configs)}: initial_lr={initial_lr:g}, meta_lr={meta_lr:g}", flush=True)

            def report(phase, step, steps, models, optimizers, losses):
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
                      f"Rest gesamt ~{duration(elapsed / completed * (planned - completed))}", flush=True)
                for name, model in models.items():
                    weights = model.weight.detach().flatten()
                    state = optimizers[name].state[model.weight]
                    rates = state["beta"].exp().flatten() if name == "IDBD" else torch.full_like(weights, initial_lr)
                    rms = weights[1:].square().mean().sqrt().item() if weights.numel() > 1 else 0.0
                    lr_mean = rates[1:].mean().item() if rates.numel() > 1 else rates[0].item()
                    print(f"{name}: loss={losses[name]:.5g}, w[0]={weights[0].item():.5g}, "
                          f"distractor_rms={rms:.5g}, lr[0]={rates[0].item():.5g}, "
                          f"distractor_lr={lr_mean:.5g}", flush=True)
                last_log = time.perf_counter()

            try:
                rows = run_config(args, initial_lr, meta_lr, report)
                status, error = "ok", ""
            except FloatingPointError as exc:
                status, error = "diverged", str(exc)
                rows = [dict(model="IDBD"), dict(model="SGD")]
                planned -= per_run - run_done
                print(f"Lauf abgebrochen: {error}; weiter mit nächster Kombination.", flush=True)
            seconds = time.perf_counter() - run_start
            for row in rows:
                writer.writerow(dict(initial_lr=initial_lr, meta_lr=meta_lr, seed=args.seed,
                                     status=status, seconds=seconds, error=error, **row))
                if status == "ok":
                    active_mse = row["active_clean_mse"]
                    active_text = f"{active_mse:.6g}" if active_mse != "" else "n/a"
                    print(f"Test {row['model']}: clean_mse={row['clean_mse']:.6g}, "
                          f"noisy_mse={row['noisy_mse']:.6g}, active_clean_mse={active_text}", flush=True)
            output.flush()
    print(f"\nFertig nach {duration(time.perf_counter() - start)}. Ergebnisse: {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen; bereits abgeschlossene Läufe stehen in der CSV.", flush=True)
        raise SystemExit(130)
