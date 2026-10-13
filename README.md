# SGD vs. IDBD

Compare SGD with Incremental Delta-Bar-Delta on a reproducible stream of sparse
binary features and noisy targets. Run commands from the repository root.

## Layout

- `src/optim/IDBD.py`: PyTorch optimizer implementation; import with
  `from src.optim import IDBD`.
- `src/optim/IDBDMomentum.py`: `IDBDMomentum`, with heavy-ball momentum only
  on log learning rates (beta), not on weights.
- `src/data.py`: infinite `DataStream` dataset; the first feature is the clean
  target, with Gaussian noise and occasional impulses added to the noisy target.
- `src/main.py`: shared `train()`, `evaluate()` and metric APIs; single comparison
  with MLflow metrics, plots and CSV artifacts.
- `src/sweep.py`: parameter grid using those same APIs; one parent MLflow run,
  one nested run per configuration, and a summary CSV.
- `src/test.py`: input-dimension / stopping-condition experiment.
- `notebooks/`: interactive experiments using the shared optimizer and dataset.
- `exercises/idbd_exercise.py`: compatibility import, not a second implementation.

## Beta-momentum notebook

Open `notebooks/idbd_momentum.ipynb` using this project's Python environment and
run all cells from the repository root or `notebooks/`. It compares SGD, IDBD
and IDBDMomentum on identical initial weights and batches, using `src.main.train()`,
`evaluate()` and `learning_rates()` without MLflow (`log_mlflow=False`).
Plots show distractor weight RMS, individual distractor weights/rates, learning
rate quantiles and final distributions, useful-feature behavior and independent
clean-target test errors. Settings are in the configuration cell; history is
sampled every `RECORD_EVERY` updates to bound memory usage.

`IDBDMomentum` uses `v = beta_momentum * v - grad * h`, then
`beta += meta_lr * v`; the original IDBD weight and h updates are unchanged.
Import it with `from src.optim import IDBDMomentum` and call `step(x)` after
backpropagating mean half-squared error, just as for IDBD. `beta_momentum`
defaults to 0.9 and must be in [0, 1); setting it to 0 recovers IDBD.
Buffers are per weight and saved in the optimizer state dict.

The main notebook comparison uses equal meta learning rates. Unnormalized
momentum amplifies a constant meta-gradient by `1 / (1 - beta_momentum)` in the
long run. Enable `RUN_MATCHED_META_LR` for a second comparison that reduces only
the momentum variant's meta learning rate by `1 - beta_momentum`. This separates
long-run amplification from the memory effect, but does not equate transients.
Faster rate shrinkage is not itself evidence of better weights or test error.

## Shared training API

`train(dataset, iterations, batch_size, models, optimizers, ...)` accepts named
model and optimizer dictionaries with identical, nonempty keys. Create and
initialize models outside training, then bind each optimizer to its corresponding
model's parameters. The experiment expects scalar linear predictors without bias,
with feature 0 useful and at least one distractor. Names are arbitrary except
`Zero`, reserved for the automatic baseline. IDBD subclasses (including
IDBDMomentum) use `step(x)`; other PyTorch optimizers use `step()`.

`create_comparison(input_dim, initial_lr, meta_lr, zeros=False)` builds the default
IDBD/SGD pair with identical initial weights (`zeros` is keyword-only). Custom
comparisons can supply their own dictionaries. Learning rates and initialization
are no longer arguments of `train()`. Its five return values remain
`models, optimizers, history, train_metrics, predictions`; the first two are the
supplied dictionaries, updated in place, so training can resume with existing
optimizer state. `stop_condition` waits until all supplied models meet the target.
Use `log_mlflow=False` without tracking; importing `src.main` leaves the plotting
backend unchanged.

## Single comparison

```sh
uv run python -m src.main --input-dim 16 --iterations 1000
```

## Sweep

```sh
uv run python -m src.sweep --input-dim 16 --iterations 1000 --initial-lrs 0.001 0.01 --meta-lrs 0 0.01 --output sweep_results.csv
```

Each configuration starts from the same random weights and the same fresh training
and test streams. Both models share their initial weights. Use `--zeros` to retain
the previous sweep's zero initialization. `--steps` and `--test-steps` remain aliases
for `--iterations` and `--test-iterations`.

The CSV contains one row per configuration and model (IDBD, SGD, Zero), including
train/test MSEs, active-only test MSE, final weight/rate summaries, run ID, duration
and status. Undefined active-only MSEs are empty. The CSV is also saved as an
artifact of the parent run. Nested runs contain training/test metrics and metric
CSV artifacts. Divergent configurations are marked FAILED and do not stop the
sweep; their CSV rows contain the error instead of metrics.

Sweep history collection is disabled to keep memory usage bounded. The default
`train()` API still returns full history and recent predictions. Its optional
`progress_callback(step, metrics_by_model)` reports every update; `evaluate()`
supports the same callback for test batches.

Direct script execution also works: `uv run python src/main.py` or
`uv run python src/sweep.py`.

## MLflow

The default tracking URI is `http://100.96.85.78:5000`. The tracking server must
be reachable during training; open the same URL to view runs in the MLflow UI.
`src.main` and `src.sweep` accept `--tracking-uri` and `--experiment` overrides.
The input-dimension script and notebook also use this tracking server.
Local tracking databases and artifacts (`mlflow.db*`, `mlruns/`, `mlartifacts/`)
are ignored by Git and are not kept as repository snapshots.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```sh
git clone git@github.com:jasperjonkhans/IDBD-optimizer.git
cd IDBD-optimizer
uv sync --locked
```

`uv` installs the Python version selected in `.python-version` (Python 3.14) and dependencies from `uv.lock`. Do not copy another contributor's `.venv`.

## View the shared MLflow runs

The repository contains a compressed SQLite snapshot in `tracking/mlflow.db.gz` and the corresponding CSV/plot artifacts in `mlruns/`. Import once after cloning:

```sh
uv run python scripts/mlflow_snapshot.py import
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and select `SGD-vs-IDBD`. You can inspect parameters, metric histories, CSVs and plots. The import resolves artifact paths to your checkout, so paths from the author's computer are not required. Keep the checkout in place while viewing runs.

The snapshot captures run statuses at export time, including failed or still-running runs; it is not a live tracking service. Runs without logged artifacts have no artifact files to display.

Import refuses to overwrite an existing database. To view an updated snapshot without losing your own runs:

```sh
uv run python scripts/mlflow_snapshot.py import --database shared.db
uv run mlflow ui --backend-store-uri sqlite:///shared.db --host 127.0.0.1 --port 5001
```

Use a new database filename for each import, or explicitly back up and remove an old local copy first. Local databases are ignored by Git.

## Tests

`uv run python -m src.test` runs the longer input-dimension stopping-condition
experiment, not the unit tests.

Unit tests:

```sh
uv run python -m unittest discover -s tests -v
```

## Share updated results

After training has finished, stop any writers before exporting and committing. Export uses SQLite's backup API and does not modify your live database:

```sh
uv run python scripts/mlflow_snapshot.py export
git add tracking/mlflow.db.gz mlruns/
git commit -m "Update shared MLflow runs"
git push origin main
```

The snapshot contains all database runs/metrics/parameters/tags. Review them and all artifacts before publishing; do not log credentials or sensitive data. The exporter replaces machine-specific artifact paths with portable paths. Other recorded metadata, such as user names and source tags, is preserved.

Snapshot and artifacts must be published together. GitHub rejects individual files over 100 MiB; the exporter refuses compressed snapshots at 95 MiB. If results grow beyond that, switch to Git LFS or a shared tracking server. The snapshot is a full copy, not a merge mechanism for contributor databases.
