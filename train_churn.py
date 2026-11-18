"""Train and evaluate a churn classifier inside the SageMaker training container."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def load_dataset(directory, filename):
    """Load a headerless churn CSV and split its labels from its features."""
    df = pd.read_csv(Path(directory) / filename, header=None)
    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return X, y


def train_decision_tree(X_train, y_train, args):
    """Fit and return a decision-tree classifier using the provided settings."""
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.seed)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, args):
    """Fit and return an XGBoost classifier using the provided settings."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.num_round,
        learning_rate=args.eta,
        objective="binary:logistic",
        random_state=args.seed,
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_validation, y_validation):
    """Calculate classification metrics for the validation dataset."""
    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]
    return {
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(y_validation, predictions, zero_division=0),
        "recall": recall_score(y_validation, predictions, zero_division=0),
        "f1": f1_score(y_validation, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_validation, probabilities),
        "average_precision": average_precision_score(y_validation, probabilities),
    }


def save_model(model, algorithm, model_dir, metrics):
    """Save the trained model and its evaluation metrics to the model directory."""
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if algorithm == "tree":
        import joblib

        joblib.dump(model, output_dir / "model.joblib")
    else:
        model.save_model(str(output_dir / "xgboost-model.json"))
    # SageMaker automatically packages /opt/ml/model and uploads it to S3.
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def parse_arguments():
    """Parse command-line and SageMaker environment training settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["tree", "xgboost"], required=True)
    parser.add_argument(
        "--train", default=os.getenv("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    )
    parser.add_argument(
        "--validation",
        default=os.getenv("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"),
    )
    parser.add_argument(
        "--model-dir", default=os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-round", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.2)
    return parser.parse_args()


def main():
    """Load data, train and evaluate the selected model, then save its artifacts."""
    args = parse_arguments()
    X_train, y_train = load_dataset(args.train, "train.csv")
    X_validation, y_validation = load_dataset(args.validation, "validation.csv")

    if args.algorithm == "tree":
        model = train_decision_tree(X_train, y_train, args)
    else:
        model = train_xgboost(X_train, y_train, args)

    metrics = evaluate_model(model, X_validation, y_validation)
    # SageMaker extracts these values from CloudWatch logs via metric definitions.
    for name, value in metrics.items():
        print(f"validation_{name}={value:.8f}", flush=True)
    save_model(model, args.algorithm, args.model_dir, metrics)


if __name__ == "__main__":
    main()
