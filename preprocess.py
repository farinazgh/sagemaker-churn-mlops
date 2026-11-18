"""Clean, split and encode the churn dataset locally or in SageMaker."""

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
SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace("’", "'", regex=False)
    expected = set(CAT + NUM + ["Phone", TARGET])
    if set(df.columns) != expected or df.columns.duplicated().any():
        raise ValueError("Unexpected dataset columns")
    if df.isna().any().any() or df.duplicated().any():
        raise ValueError("Dataset contains missing values or duplicate rows")

    # Like Spark StringIndexer, but explicitly map the binary target.
    labels = df[TARGET].astype(str).str.strip().str.lower().str.rstrip(".")
    if set(labels.unique()) != {"true", "false"}:
        raise ValueError("Expected True./False. churn labels")
    df[TARGET] = labels.map({"true": 1, "false": 0})

    for column in CAT:
        df[column] = df[column].astype(str).str.strip()
        if (df[column] == "").any():
            raise ValueError(f"Empty category: {column}")
    for column in ["Int'l Plan", "VMail Plan"]:
        df[column] = df[column].str.lower()
        if not df[column].isin(["yes", "no"]).all():
            raise ValueError(f"Expected yes/no: {column}")
    df[NUM] = df[NUM].apply(pd.to_numeric)
    if not np.isfinite(df[NUM].to_numpy()).all() or (df[NUM] < 0).any().any():
        raise ValueError("Invalid numeric values")

    # A customer must not occur in different splits.
    df["Phone"] = df["Phone"].astype(str).str.strip()
    if df.duplicated(["Area Code", "Phone"]).any():
        raise ValueError("Repeated customer identifiers")
    return df.drop(columns="Phone")


def split_data(df):
    # Stratification preserves the churn ratio in each independent split.
    train, holdout = train_test_split(
        df, test_size=0.4, random_state=SEED, stratify=df[TARGET]
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, random_state=SEED, stratify=holdout[TARGET]
    )
    return train, validation, test


def fit_schema(train):
    # Learn categories from training data only, not validation or test.
    categories = {column: sorted(train[column].unique()) for column in CAT}
    feature_columns = NUM.copy()
    for column, values in categories.items():
        feature_columns.extend(f"{column}={value}" for value in values)
    return {
        "schema_version": 1,
        "target": TARGET,
        "label_first": True,
        "header": False,
        "numeric_columns": NUM,
        "categories": categories,
        "feature_columns": feature_columns,
        "unknown_category_policy": "all zero within that categorical feature",
    }


def transform(df, schema):
    # Equivalent to fitting a Spark encoder on train and applying it to all splits.
    features = df[schema["numeric_columns"]].astype(float).copy()
    unknown = {}
    for column, values in schema["categories"].items():
        unknown[column] = int((~df[column].isin(values)).sum())
        for value in values:
            features[f"{column}={value}"] = (df[column] == value).astype(int)
    # SageMaker XGBoost expects the target in column zero.
    return pd.concat([df[TARGET], features[schema["feature_columns"]]], axis=1), unknown


def write_datasets(train, validation, test, schema, output_dir):
    splits = {}
    for name, df in [("train", train), ("validation", validation), ("test", test)]:
        encoded, unknown = transform(df, schema)
        path = output_dir / name / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded.to_csv(path, header=False, index=False)
        splits[name] = {
            "rows": len(df),
            "churn_rate": float(df[TARGET].mean()),
            "sha256": sha256(path),
            "unknown_categories": unknown,
        }
        print(f"{name}: {encoded.shape}, churn rate: {df[TARGET].mean():.4f}")
    return splits


def run(input_file, output_dir):
    df = load_data(input_file)
    train, validation, test = split_data(df)
    schema = fit_schema(train)
    output_dir = Path(output_dir)
    splits = write_datasets(train, validation, test, schema, output_dir)

    report = {
        "raw_sha256": sha256(input_file),
        "rows": len(df),
        "seed": SEED,
        "raw_churn_rate": float(df[TARGET].mean()),
        "feature_count": len(schema["feature_columns"]),
        "dropped_columns": ["Phone"],
        "splits": splits,
        "versions": {"pandas": pd.__version__},
    }
    metadata = output_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "feature_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    (metadata / "data_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.input_file, args.output_dir)
