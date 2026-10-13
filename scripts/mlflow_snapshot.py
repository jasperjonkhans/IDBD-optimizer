"""Share a consistent local MLflow SQLite snapshot (standard library only)."""

import argparse
import gzip
from pathlib import Path
import shutil
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tracking" / "mlflow.db.gz"
DATABASE = ROOT / "mlflow.db"


def relocate_artifacts(connection, root=None):
    """Only rewrite artifact locations; keep run IDs, metrics and tags unchanged."""
    for experiment_id, in connection.execute("SELECT experiment_id FROM experiments").fetchall():
        path = Path("mlruns") / str(experiment_id)
        location = (root / path).as_uri() if root else path.as_posix()
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
            (location, experiment_id),
        )
    for run_id, experiment_id in connection.execute(
        "SELECT run_uuid, experiment_id FROM runs"
    ).fetchall():
        path = Path("mlruns") / str(experiment_id) / run_id / "artifacts"
        location = (root / path).as_uri() if root else path.as_posix()
        connection.execute("UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                           (location, run_id))
    connection.commit()


def export_snapshot(database, snapshot):
    if not database.is_file():
        raise FileNotFoundError(database)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=snapshot.parent) as directory:
        backup = Path(directory) / "mlflow.db"
        # SQLite backup also includes committed WAL contents, without editing the source.
        with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as source:
            with sqlite3.connect(backup) as target:
                source.backup(target)
                relocate_artifacts(target)
        compressed = Path(directory) / "mlflow.db.gz"
        with backup.open("rb") as source, compressed.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as target:
                shutil.copyfileobj(source, target)
        if compressed.stat().st_size >= 95 * 1024 * 1024:
            raise ValueError("Snapshot too large for GitHub; use LFS or external storage")
        compressed.replace(snapshot)
    print(f"Exported {snapshot} ({snapshot.stat().st_size / 1024**2:.1f} MiB)")


def import_snapshot(snapshot, database, root=ROOT):
    if database.exists():
        raise FileExistsError(f"Refusing to overwrite {database}; use --database with a new path")
    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=database.parent) as directory:
        restored = Path(directory) / "mlflow.db"
        with gzip.open(snapshot, "rb") as source, restored.open("wb") as target:
            shutil.copyfileobj(source, target)
        with sqlite3.connect(restored) as connection:
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Invalid SQLite snapshot")
            relocate_artifacts(connection, root.resolve())
        # Exclusive creation protects an existing DB, even if created during import.
        with restored.open("rb") as source, database.open("xb") as target:
            shutil.copyfileobj(source, target)
    print(f"Imported {database}; artifacts resolved against {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "import"))
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    try:
        if args.action == "export":
            export_snapshot(args.database, args.snapshot)
        else:
            import_snapshot(args.snapshot, args.database)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
