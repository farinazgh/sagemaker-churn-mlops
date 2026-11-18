"""Local tests: no AWS calls or compute charges."""

import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from train_churn import load_dataset

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "run_training", ROOT / "03-run_training.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for name, start, count in [("train", 0, 60), ("validation", 60, 20)]:
            folder = self.root / name
            folder.mkdir()
            rows = [[i % 2, i % 2, i / 100] for i in range(start, start + count)]
            pd.DataFrame(rows).to_csv(folder / f"{name}.csv", header=False, index=False)

    def tearDown(self):
        self.directory.cleanup()

    def train_and_verify(self, algorithm):
        output = self.root / algorithm
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "train_churn.py"),
                "--algorithm",
                algorithm,
                "--train",
                str(self.root / "train"),
                "--validation",
                str(self.root / "validation"),
                "--model-dir",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads((output / "metrics.json").read_text())
        self.assertEqual(report["algorithm"], algorithm)
        self.assertEqual(report["train_rows"], 60)
        self.assertEqual(report["validation_rows"], 20)
        self.assertEqual(report["feature_count"], 2)
        self.assertGreater(report["validation"]["f1"], 0.9)
        artifact = self.root / "model.tar.gz"
        with tarfile.open(artifact, "w:gz") as archive:
            for file in output.iterdir():
                archive.add(file, arcname=file.name)
        self.assertEqual(runner.read_and_verify_model_metrics(artifact), report)
        return report

    def test_decision_tree_training(self):
        self.train_and_verify("tree")

    def test_xgboost_training(self):
        self.train_and_verify("xgboost")

    def test_wrong_input_checksum(self):
        with self.assertRaises(ValueError):
            load_dataset(self.root / "train", "train", "incorrect")

    def test_invalid_labels(self):
        pd.DataFrame([[3, 0], [0, 1]]).to_csv(
            self.root / "train/train.csv", index=False, header=False
        )
        with self.assertRaises(ValueError):
            load_dataset(self.root / "train", "train")

    def test_different_training_data(self):
        run = {
            "processing_job": "old",
            "data_hashes": {"train": "old", "validation": "old"},
        }
        manifest = {"job_name": "new"}
        report = {
            "splits": {"train": {"sha256": "new"}, "validation": {"sha256": "new"}}
        }
        with self.assertRaises(ValueError):
            runner.validate_existing_run_data(run, manifest, report)

    def test_corrupt_model_artifact(self):
        output = self.root / "broken"
        output.mkdir()
        (output / "model.joblib").write_bytes(b"corrupt")
        (output / "metrics.json").write_text(
            json.dumps({"model_file": "model.joblib", "model_sha256": "invalid"})
        )
        artifact = self.root / "broken.tar.gz"
        with tarfile.open(artifact, "w:gz") as archive:
            for file in output.iterdir():
                archive.add(file, arcname=file.name)
        with self.assertRaises(ValueError):
            runner.read_and_verify_model_metrics(artifact)


if __name__ == "__main__":
    unittest.main()
