import gzip
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.mlflow_snapshot import export_snapshot, import_snapshot


class SnapshotTests(unittest.TestCase):
    def test_round_trip_relocates_artifacts_without_changing_source_or_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.db"
            snapshot = root / "tracking" / "mlflow.db.gz"
            destination = root / "restored.db"
            with sqlite3.connect(source) as connection:
                connection.executescript("""
                    CREATE TABLE experiments (experiment_id INTEGER, artifact_location TEXT);
                    CREATE TABLE runs (run_uuid TEXT, experiment_id INTEGER, artifact_uri TEXT);
                    CREATE TABLE metrics (key TEXT, value REAL);
                    INSERT INTO experiments VALUES (1, '/old/checkout/mlruns/1');
                    INSERT INTO runs VALUES ('abc', 1, '/old/checkout/mlruns/1/abc/artifacts');
                    INSERT INTO metrics VALUES ('loss', 0.125);
                """)
            export_snapshot(source, snapshot)
            with sqlite3.connect(source) as connection:
                self.assertEqual(connection.execute("SELECT artifact_uri FROM runs").fetchone()[0],
                                 "/old/checkout/mlruns/1/abc/artifacts")
            import_snapshot(snapshot, destination, root)
            with sqlite3.connect(destination) as connection:
                self.assertEqual(connection.execute("SELECT artifact_uri FROM runs").fetchone()[0],
                                 (root / "mlruns/1/abc/artifacts").as_uri())
                self.assertEqual(connection.execute("SELECT artifact_location FROM experiments").fetchone()[0],
                                 (root / "mlruns/1").as_uri())
                self.assertEqual(connection.execute("SELECT value FROM metrics").fetchone()[0], 0.125)
            previous = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                import_snapshot(snapshot, destination, root)
            self.assertEqual(destination.read_bytes(), previous)

    def test_invalid_snapshot_does_not_create_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "bad.gz"
            with gzip.open(snapshot, "wb") as output:
                output.write(b"not a database")
            with self.assertRaises(sqlite3.DatabaseError):
                import_snapshot(snapshot, root / "restored.db", root)
            self.assertFalse((root / "restored.db").exists())


if __name__ == "__main__":
    unittest.main()
