"""Reproducible, leakage-aware preprocessing for the AWS churn.txt sample."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Churn?"
CAT = ["State", "Area Code", "Int'l Plan", "VMail Plan"]
NUM = [
    "Account Length",
    "VMail Message",
    "Day Mins",
    "Day Calls",
    "Day Charge",
    "Eve Mins",
    "Eve Calls",
    "Eve Charge",
    "Night Mins",
    "Night Calls",
    "Night Charge",
    "Intl Mins",
    "Intl Calls",
    "Intl Charge",
    "CustServ Calls",
]
REQUIRED = set(CAT + NUM + ["Phone", TARGET])
SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("’", "'", regex=False)
    if set(df.columns) != REQUIRED or df.columns.duplicated().any():
        raise ValueError(
            f"Schema mismatch: missing={REQUIRED-set(df.columns)}, extra={set(df.columns)-REQUIRED}"
        )
    if df.isna().any().any():
        raise ValueError(
            f"Missing values: {df.isna().sum().loc[lambda x:x>0].to_dict()}"
        )
    if df.duplicated().any():
        raise ValueError("Duplicate raw rows; resolve before splitting.")
    label = df[TARGET].astype(str).str.strip().str.lower().str.rstrip(".")
    if set(label.unique()) != {"true", "false"}:
        raise ValueError("Expected exactly true/false labels.")
    df[TARGET] = label.map({"true": 1, "false": 0}).astype(int)
    for col in CAT:
        df[col] = df[col].astype(str).str.strip()
        if (df[col] == "").any():
            raise ValueError(f"Empty categorical value: {col}")
    for col in ["Int'l Plan", "VMail Plan"]:
        df[col] = df[col].str.lower()
        if not set(df[col].unique()) <= {"yes", "no"}:
            raise ValueError(f"Unexpected values in {col}")
    for col in NUM:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"Nonfinite/negative numeric data in {col}")
    # Drop the identifier before modeling. Check repeated identifiers first so
    # observations of a single customer cannot land in different splits.
    if (
        df.assign(Phone=df["Phone"].astype(str).str.strip())
        .duplicated(subset=["Area Code", "Phone"])
        .any()
    ):
        raise ValueError(
            "Repeated area-code/phone identifiers: investigate or use group-wise splitting."
        )
    df = df.drop(columns="Phone")
    return df


def fit_schema(train):
    categories = {c: sorted(train[c].unique().tolist()) for c in CAT}
    columns = NUM + [f"{c}={v}" for c in CAT for v in categories[c]]
    return {
        "schema_version": 1,
        "target": TARGET,
        "label_first": True,
        "header": False,
        "numeric_columns": NUM,
        "categories": categories,
        "feature_columns": columns,
        "unknown_category_policy": "all zero within that categorical feature",
    }


def transform(df, schema):
    features = {c: df[c].astype(float) for c in schema["numeric_columns"]}
    unknown = {}
    for c, values in schema["categories"].items():
        unknown[c] = int((~df[c].isin(values)).sum())
        for value in values:
            features[f"{c}={value}"] = (df[c] == value).astype(int)
    x = pd.DataFrame(features, index=df.index)[schema["feature_columns"]]
    return pd.concat([df[TARGET], x], axis=1), unknown


def run(input_file, output_dir):
    df = load_data(input_file)
    train, holdout = train_test_split(
        df, test_size=0.4, random_state=SEED, stratify=df[TARGET]
    )
    val, test = train_test_split(
        holdout, test_size=0.5, random_state=SEED, stratify=holdout[TARGET]
    )
    assert not (
        set(train.index) & set(val.index)
        or set(train.index) & set(test.index)
        or set(val.index) & set(test.index)
    )
    schema = fit_schema(train)
    out = Path(output_dir)
    report = {
        "raw_sha256": sha256(input_file),
        "rows": len(df),
        "seed": SEED,
        "raw_churn_rate": float(df[TARGET].mean()),
        "feature_count": len(schema["feature_columns"]),
        "dropped_columns": ["Phone"],
        "splits": {},
        "versions": {"pandas": pd.__version__},
    }
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        encoded, unknown = transform(part, schema)
        dest = out / name / f"{name}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        encoded.to_csv(dest, index=False, header=False)
        report["splits"][name] = {
            "rows": len(part),
            "churn_rate": float(part[TARGET].mean()),
            "sha256": sha256(dest),
            "unknown_categories": unknown,
        }
    meta = out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    (meta / "data_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    run(a.input_file, a.output_dir)
