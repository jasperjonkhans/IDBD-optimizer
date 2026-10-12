"""SGD/IDBD comparison from the notebook, with local MLflow tracking."""

import argparse
import copy
import math
from collections import deque
from itertools import islice

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from helper import DataStream
from idbd_exercise import IDBD

plt.switch_backend("Agg")  # Save plots without opening windows (also works headlessly).


def metric_totals():
    return dict(examples=0, active_examples=0, noisy_sse=0.0,
                clean_sse=0.0, active_clean_sse=0.0)


def accumulate_metrics(totals, prediction, noisy_y, clean_y):
    prediction = prediction.detach().double()
    noisy_error = prediction - noisy_y.double()
    clean_error = prediction - clean_y.double()
    active = clean_y == 1
    totals["examples"] += clean_y.numel()
    totals["active_examples"] += active.sum().item()
    totals["noisy_sse"] += noisy_error.square().sum().item()
    totals["clean_sse"] += clean_error.square().sum().item()
    totals["active_clean_sse"] += clean_error[active].square().sum().item()


def summarize_metrics(totals_by_model):
    rows = []
    for name, totals in totals_by_model.items():
        count = totals["examples"]
        if count == 0:
            raise ValueError("The data stream produced no examples")
        active_count = totals["active_examples"]
        noisy_mse = totals["noisy_sse"] / count
        rows.append({
            "model": name,
            "examples": count,
            "active_examples": active_count,
            "half_squared_loss": 0.5 * noisy_mse,
            "noisy_mse": noisy_mse,
            "clean_mse": totals["clean_sse"] / count,
            "active_clean_mse": (
                totals["active_clean_sse"] / active_count
                if active_count else float("nan")
            ),
        })
    return pd.DataFrame(rows).set_index("model")


def log_summary(metrics, prefix, step):
    # Active-only MSE is undefined until at least one active example was seen.
    mlflow.log_metrics({
        f"{prefix}.{name}.{key}": float(value)
        for name, row in metrics.iterrows()
        for key, value in row.items()
        if math.isfinite(value)
    }, step=step)


def learning_rates(model, optimizer):
    if isinstance(optimizer, IDBD):
        return optimizer.state[model.weight]["beta"].exp().squeeze(0)
    return torch.full_like(model.weight.squeeze(0), optimizer.param_groups[0]["lr"])


def train(dataset, iterations, batch_size, initial_lr, meta_lr, log_every=100,
          stop_condition=None, zeros=False):
    """Train for at most ``iterations`` updates on the shared data stream.

    If provided, ``stop_condition(step, name, metrics)`` is checked after each
    optimizer update until its first True result. ``metrics`` contains the same
    values as a history row: pre-update losses and post-update weights/rates.
    Both models continue training until both have reached the criterion.
    The returned training metrics then include ``steps_to_target``: the first
    matching update, or None if not reached (also None for the Zero baseline).
    Without a callback, the return values and fixed-budget behavior are unchanged.
    """
    # Same random initial weights for both models, as in the notebook's train().
    idbd_model = nn.Linear(dataset.input_dim, 1, bias=False)
    if zeros:
        nn.init.zeros_(idbd_model.weight)
    models = {"IDBD": idbd_model, "SGD": copy.deepcopy(idbd_model)}
    optimizers = {
        "IDBD": IDBD(models["IDBD"].parameters(),
                     initial_lr=initial_lr, meta_lr=meta_lr),
        "SGD": torch.optim.SGD(models["SGD"].parameters(), lr=initial_lr),
    }
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    totals = {name: metric_totals() for name in (*models, "Zero")}
    history = []
    last_predictions = deque(maxlen=500)
    recent_losses = {name: deque(maxlen=200) for name in models}
    examples_seen = 0
    next_report = 10
    steps_to_target = {name: None for name in models}

    for step, (x, noisy_y, clean_y) in enumerate(islice(loader, iterations), start=1):
        predictions = {}
        step_metrics = {}
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x).squeeze(-1)
            loss = 0.5 * (prediction - noisy_y).square().mean()
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"{name}: non-finite loss at update {step}")

            predictions[name] = prediction.detach().clone()
            accumulate_metrics(totals[name], prediction, noisy_y, clean_y)
            loss.backward()
            if name == "IDBD":
                optimizer.step(x)
            else:
                optimizer.step()

            with torch.no_grad():
                weights = model.weight.squeeze(0)
                rates = learning_rates(model, optimizer)
                finite_h = (torch.isfinite(optimizer.state[model.weight]["h"]).all().item()
                            if name == "IDBD" else True)
                if not (torch.isfinite(weights).all().item()
                        and torch.isfinite(rates).all().item() and finite_h):
                    raise FloatingPointError(
                        f"{name}: non-finite optimizer state at update {step}"
                    )
                recent_losses[name].append(loss.item())
                values = {
                    "loss": loss.item(),
                    "loss_ma_200": sum(recent_losses[name]) / len(recent_losses[name]),
                    "useful_weight": weights[0].item(),
                    "distractor_weight_rms": weights[1:].square().mean().sqrt().item(),
                    "useful_lr": rates[0].item(),
                    "distractor_lr_mean": rates[1:].mean().item(),
                    "lr_min": rates.min().item(),
                    "lr_max": rates.max().item(),
                }
            history.append({"model": name, "update": step,
                            "examples": examples_seen + len(x), **values})
            step_metrics.update({f"train.{name}.{key}": value
                                 for key, value in values.items()})
            if (stop_condition is not None and steps_to_target[name] is None
                    and stop_condition(step, name, values.copy())):
                steps_to_target[name] = step

        accumulate_metrics(totals["Zero"], torch.zeros_like(clean_y), noisy_y, clean_y)
        for i in range(len(x)):
            last_predictions.append({
                "example": examples_seen + i + 1,
                "clean_target": clean_y[i].item(),
                "noisy_target": noisy_y[i].item(),
                **{name: prediction[i].item() for name, prediction in predictions.items()},
            })
        examples_seen += len(x)

        all_reached = (stop_condition is not None
                       and all(value is not None for value in steps_to_target.values()))
        if step == 1 or step % log_every == 0 or step == iterations or all_reached:
            mlflow.log_metrics(step_metrics, step=step)
            log_summary(summarize_metrics(totals), "train", step)
        progress = step / iterations * 100
        if progress >= next_report or step == iterations or all_reached:
            print(f"Training {progress:.0f}%: {examples_seen} examples, "
                  f"IDBD loss={step_metrics['train.IDBD.loss']:.4f}, "
                  f"SGD loss={step_metrics['train.SGD.loss']:.4f}")
            next_report = (int(progress) // 10 + 1) * 10
        if all_reached:
            break

    train_metrics = summarize_metrics(totals)
    if stop_condition is not None:
        train_metrics["steps_to_target"] = pd.Series(
            {**steps_to_target, "Zero": None}, dtype=object
        )
    return (models, optimizers, pd.DataFrame(history), train_metrics,
            pd.DataFrame(last_predictions))


@torch.no_grad()
def evaluate(models, dataset, iterations, batch_size):
    totals = {name: metric_totals() for name in (*models, "Zero")}
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    training_modes = {name: model.training for name, model in models.items()}
    try:
        for model in models.values():
            model.eval()
        for x, noisy_y, clean_y in islice(loader, iterations):
            for name, model in models.items():
                prediction = model(x).squeeze(-1)
                if not torch.isfinite(prediction).all().item():
                    raise FloatingPointError(f"{name}: non-finite test prediction")
                accumulate_metrics(totals[name], prediction, noisy_y, clean_y)
            accumulate_metrics(totals["Zero"], torch.zeros_like(clean_y), noisy_y, clean_y)
    finally:
        for name, model in models.items():
            model.train(training_modes[name])
    metrics = summarize_metrics(totals)
    metrics["distractor_weight_rms"] = [
        model.weight[:, 1:].square().mean().sqrt().item()
        for model in models.values()
    ] + [0.0]
    return metrics


def log_plots(models, history, predictions, test_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for name in models:
        run = history[history["model"] == name]
        axes[0, 0].plot(run["examples"], run["loss_ma_200"], label=name)
        axes[0, 1].plot(run["examples"], run["useful_weight"], label=f"{name}: useful")
        axes[0, 1].plot(run["examples"], run["distractor_weight_rms"],
                        linestyle="--", label=f"{name}: distractor RMS")
        axes[1, 0].plot(run["examples"], run["useful_lr"], label=name)
        axes[1, 1].plot(run["examples"], run["distractor_lr_mean"], label=name)
    axes[0, 0].set_title("Pre-update half-squared loss (200-update MA)")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_title("Weights after each update")
    axes[0, 1].set_ylabel("Weight / RMS")
    axes[0, 1].axhline(1, color="gray", linestyle=":", label="Ideal useful weight")
    axes[1, 0].set_title("Useful feature learning rate")
    axes[1, 1].set_title("Mean distractor learning rate")
    for ax in axes[1]:
        ax.set_yscale("log")
        ax.set_ylabel("Learning rate")
    for ax in axes.flat:
        ax.set_xlabel("Examples processed")
        ax.legend()
    mlflow.log_figure(fig, "plots/training.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, name in zip(axes, models):
        ax.plot(predictions["example"], predictions[name], alpha=0.8,
                label=f"{name}: pre-update prediction")
        ax.set_title(f"{name}: last {len(predictions)} training examples")
        ax.set_ylabel("Target / prediction")
        ax.legend()
    axes[-1].set_xlabel("Example")
    mlflow.log_figure(fig, "plots/predictions.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, column, title in zip(
        axes, ("noisy_mse", "clean_mse", "active_clean_mse", "distractor_weight_rms"),
        ("Noisy target MSE", "Clean target MSE", "Clean MSE when x[0] = 1",
         "Distractor weight RMS"),
    ):
        ax.bar(test_metrics.index, test_metrics[column])
        ax.set_title(title)
        ax.set_ylabel("RMS weight" if column == "distractor_weight_rms" else "MSE")
    mlflow.log_figure(fig, "plots/test_metrics.png")
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2**15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--initial-lr", type=float, default=0.01)
    parser.add_argument("--meta-lr", type=float, default=0.1)
    parser.add_argument("--test-iterations", type=int, default=100)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log metrics every N updates; CSVs/plots include every update")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="SGD-vs-IDBD")
    args = parser.parse_args(argv)
    if args.input_dim < 2:
        parser.error("--input-dim must be at least 2 (useful feature + distractors)")
    for key in ("iterations", "batch_size", "test_iterations", "test_batch_size", "log_every"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not math.isfinite(args.initial_lr) or args.initial_lr <= 0:
        parser.error("--initial-lr must be finite and positive")
    if not math.isfinite(args.meta_lr) or args.meta_lr < 0:
        parser.error("--meta-lr must be finite and non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    # One run contains the full comparison; no server needed during training.
    with mlflow.start_run() as run:
        params = vars(args).copy()
        params.update(test_seed=args.seed + 1, loss="mean half-squared error",
                      initialization="shared random weights", bias=False,
                      feature_probability=0.01, impulse_probability=0.01,
                      noise_variance=5, loss_ma_window=200, prediction_window=500)
        mlflow.log_params(params)
        models, optimizers, history, train_metrics, predictions = train(
            DataStream(input_dim=args.input_dim, seed=args.seed),
            args.iterations, args.batch_size, args.initial_lr, args.meta_lr, args.log_every,
        )
        test_metrics = evaluate(
            models, DataStream(input_dim=args.input_dim, seed=args.seed + 1),
            args.test_iterations, args.test_batch_size,
        )
        log_summary(test_metrics, "test", args.iterations)
        log_plots(models, history, predictions, test_metrics)

        # Save complete histories and every final weight/rate as plain CSVs.
        final_parameters = pd.DataFrame({"feature": range(args.input_dim)})
        for name, model in models.items():
            final_parameters[f"{name}_weight"] = model.weight.detach().squeeze(0).numpy()
            final_parameters[f"{name}_lr"] = learning_rates(model, optimizers[name]).numpy()
        for filename, table in {
            "train_history": history, "train_metrics": train_metrics,
            "test_metrics": test_metrics, "final_predictions": predictions,
            "final_parameters": final_parameters,
        }.items():
            mlflow.log_text(table.to_csv(index=table.index.name == "model"),
                            f"data/{filename}.csv")
        print("\nTraining metrics (pre-update predictions):")
        print(train_metrics.to_string())
        print("\nTest metrics (independent stream, no updates):")
        print(test_metrics.to_string())
        print(f"\nMLflow run: {run.info.run_id}")
    return run.info.run_id


if __name__ == "__main__":
    main()
