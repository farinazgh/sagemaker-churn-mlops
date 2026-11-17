"""No AWS calls. Synthetic fixture only, not the actual AWS sample."""

import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from preprocess import CAT, NUM, TARGET, fit_schema, transform, run, load_data


def fixture(n=100):
    rows = []
    for i in range(n):
        row = {c: i + 1 for c in NUM}
        row.update(
            {
                "State": ["NY", "CA", "WA"][i % 3],
                "Area Code": [415, 510, 408][i % 3],
                "Int'l Plan": "yes" if i % 4 == 0 else "no",
                "VMail Plan": "yes" if i % 2 == 0 else "no",
                "Phone": f"555-{i:04d}",
                TARGET: "True." if i % 5 == 0 else "False.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class TestPreprocessing(unittest.TestCase):
    def test_splits_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            f = p / "churn.txt"
            fixture().to_csv(f, index=False)
            a = run(f, p / "a")
            b = run(f, p / "b")
            self.assertEqual(a, b)
            self.assertEqual(
                [a["splits"][n]["rows"] for n in ["train", "validation", "test"]],
                [60, 20, 20],
            )
            schema = json.loads((p / "a/metadata/feature_schema.json").read_text())
            self.assertNotIn("Phone", schema["feature_columns"])
            sets = []
            for name in ["train", "validation", "test"]:
                data = pd.read_csv(p / f"a/{name}/{name}.csv", header=None)
                self.assertEqual(data.shape[1], 1 + len(schema["feature_columns"]))
                self.assertEqual(data[0].mean(), 0.2)
                sets.append(set(data[1]))
            self.assertFalse(
                sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]
            )

    def test_unknown_category(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.csv"
            fixture().to_csv(p, index=False)
            df = load_data(p)
            schema = fit_schema(df)
            df.loc[0, "State"] = "NEW"
            encoded, unknown = transform(df, schema)
            self.assertEqual(unknown["State"], 1)
            self.assertEqual(
                encoded.loc[0, [c for c in encoded if c.startswith("State=")]].sum(), 0
            )

    def test_invalid_label(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.csv"
            df = fixture()
            df.loc[0, TARGET] = "maybe"
            df.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_data(p)

    def test_missing_and_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "raw.csv"
            df = fixture()
            df.loc[0, "Phone"] = df.loc[1, "Phone"]
            df.loc[0, "Area Code"] = df.loc[1, "Area Code"]
            df.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_data(p)
            df = fixture()
            df.loc[0, "State"] = None
            df.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_data(p)


if __name__ == "__main__":
    unittest.main()
