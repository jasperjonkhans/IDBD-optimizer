# IDBD optimizer

Compare SGD and Incremental Delta-Bar-Delta on a synthetic data stream. No external dataset or tracking server is required for training.

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

## Start a run

Quick smoke run:

```sh
uv run python main.py --iterations 100 --test-iterations 10
```

Full run with default settings:

```sh
uv run python main.py
```

Runs are recorded in local `mlflow.db`; plots and CSVs go into `mlruns/`. If you imported the snapshot first, your new runs appear alongside the shared ones. Available settings:

```sh
uv run python main.py --help
uv run python sweep.py --help
```

`uv run python test.py` runs the longer input-dimension stopping-condition experiment, not the unit tests. It uses MLflow's default local SQLite backend (`sqlite:///mlflow.db`); if setting `MLFLOW_TRACKING_URI`, use the same backend URI when starting the UI.

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
