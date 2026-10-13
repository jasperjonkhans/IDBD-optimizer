if __package__:
    from .main import create_comparison, train
    from .data import DataStream
else:
    from main import create_comparison, train
    from data import DataStream

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch


def main() -> None:
    mlflow.set_tracking_uri("http://100.96.85.78:5000")
    mlflow.set_experiment("input_dim_step_cnt")
    seed = 1234
    tolerance_intervall = 1e-3

    def stop_condition(step, name, values):
        return values["distractor_weight_rms"] < tolerance_intervall

    initial_lr = 0.01
    meta_lr = 0.01
    batch_size = 128
    max_iterations = 2**16
    # At least one useful feature and one distractor are needed for the RMS.
    input_dims = [2**x for x in range(1, 10)]
    results = []

    with mlflow.start_run(run_name="Steps to stop condition by input dimension"):
        mlflow.log_params({
            "seed": seed,
            "tolerance": tolerance_intervall,
            "initial_lr": initial_lr,
            "meta_lr": meta_lr,
            "batch_size": batch_size,
            "max_iterations": max_iterations,
            "input_dims": input_dims,
        })

        for input_dim in input_dims:
            with mlflow.start_run(run_name=f"input_dim={input_dim}", nested=True):
                torch.manual_seed(seed)
                mlflow.log_param("input_dim", input_dim)
                models, optimizers = create_comparison(input_dim, initial_lr, meta_lr)
                models, optimizers, history, train_metrics, predictions = train(
                    DataStream(input_dim=input_dim, seed=seed),
                    iterations=max_iterations, batch_size=batch_size,
                    models=models, optimizers=optimizers,
                    stop_condition=stop_condition,
                )

                step_count = {
                    name: train_metrics.loc[name, "steps_to_target"]
                    for name in models
                }
                for name, steps in step_count.items():
                    reached = steps is not None
                    results.append({
                        "input_dim": input_dim,
                        "model": name,
                        "steps_to_target": steps,
                        "target_reached": reached,
                    })
                    mlflow.log_metric(f"{name}.target_reached", int(reached))
                    if reached:
                        mlflow.log_metric(f"{name}.steps_to_target", int(steps))

        table = pd.DataFrame(results)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for name, rows in table.groupby("model", sort=False):
            line, = ax.plot(rows["input_dim"], rows["steps_to_target"],
                            marker="o", label=name)
            not_reached = rows[~rows["target_reached"]]
            if not not_reached.empty:
                ax.scatter(not_reached["input_dim"],
                           [max_iterations] * len(not_reached),
                           marker="x", color=line.get_color(),
                           label=f"{name}: target not reached")

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Input dimension")
        ax.set_ylabel("Optimizer updates until stop condition")
        ax.set_title(f"Distractor weight RMS < {tolerance_intervall:g}")
        ax.axhline(max_iterations, color="gray", linestyle="--",
                   label="Update limit (crosses indicate target not reached)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        try:
            mlflow.log_figure(fig, "plots/steps_to_target.png")
        finally:
            plt.close(fig)
        mlflow.log_text(table.to_csv(index=False), "data/steps_to_target.csv")


if __name__ == "__main__":
    main()
