"""Train a Decision Tree or XGBoost model using the completed Lab 02 data."""

import argparse
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
from dotenv import load_dotenv
from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.shapes import MetricDefinition, OutputDataConfig
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, SourceCode, StoppingCondition

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
LAB02_MANIFEST = ROOT / "artifacts/lab02/latest_run.json"
ARTIFACTS = ROOT / "artifacts/lab03"
INSTANCE_TYPE = os.getenv("SAGEMAKER_TRAINING_INSTANCE_TYPE", "ml.m5.large")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def resolve_training_image(algorithm, region):
    if algorithm == "tree":
        return image_uris.retrieve(
            framework="sklearn",
            version="1.4-2",
            py_version="py3",
            region=region,
            instance_type=INSTANCE_TYPE,
        )
    return image_uris.retrieve(
        framework="xgboost",
        version="3.0-5",
        image_scope="training",
        region=region,
        instance_type=INSTANCE_TYPE,
    )


def build_training_inputs(processing_run):
    output_uri = f"s3://{processing_run['bucket']}/{processing_run['prefix']}/output"
    # Lab 02 already uploaded these CSVs. Do not upload them again.
    return [
        InputData(channel_name="train", data_source=f"{output_uri}/train/train.csv"),
        InputData(
            channel_name="validation",
            data_source=f"{output_uri}/validation/validation.csv",
        ),
    ]


def train_model(session, processing_run, algorithm):
    bucket = processing_run["bucket"]
    region = session.region_name
    role = f"arn:aws:iam::{processing_run['account']}:role/SageMakerChurnLabExecutionRole-{region}"
    image = resolve_training_image(algorithm, region)

    # Package only training code, not .env or the entire repository.
    with TemporaryDirectory() as code_dir:
        shutil.copy(ROOT / "train_churn.py", code_dir)
        trainer = ModelTrainer(
            training_image=image,
            role=role,
            sagemaker_session=Session(boto_session=session, default_bucket=bucket),
            base_job_name=f"churn-l03-{algorithm}",
            source_code=SourceCode(source_dir=code_dir, entry_script="train_churn.py"),
            compute=Compute(
                instance_type=INSTANCE_TYPE, instance_count=1, volume_size_in_gb=10
            ),
            stopping_condition=StoppingCondition(max_runtime_in_seconds=1200),
            output_data_config=OutputDataConfig(
                s3_output_path=f"s3://{bucket}/lab03/runs"
            ),
            hyperparameters={"algorithm": algorithm, "max-depth": 5, "seed": 42},
        )
        if algorithm == "tree":
            trainer.with_metric_definitions([
                MetricDefinition(
                    name=f"validation:{metric}",
                    regex=rf"validation_{metric}=([0-9.]+)"
                )
                for metric in METRICS
            ])
        # 1. SDK uploads the source to S3, calls CreateTrainingJob, then waits
        #    for completion using DescribeTrainingJob and streams CloudWatch logs.
        trainer.train(
            input_data_config=build_training_inputs(processing_run),
            wait=True,
            logs=True,
        )
        # SDK v3 documents this attribute for accessing the submitted job.
        return trainer._latest_training_job.training_job_name, image


def save_training_results(session, algorithm, job_name, image, processing_run):
    sagemaker_client = session.client("sagemaker")
    # 2. DescribeTrainingJob retrieves the model artifact URI and final metrics.
    job = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    metrics = {
        metric["MetricName"].split(":", 1)[1]: metric["Value"]
        for metric in job.get("FinalMetricDataList", [])
        if metric["MetricName"].startswith("validation:")
    }
    result = {
        "algorithm": algorithm,
        "job_name": job_name,
        "image": image,
        "processing_job": processing_run["job_name"],
        "model_uri": job["ModelArtifacts"]["S3ModelArtifacts"],
        "metrics": metrics,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{algorithm}_latest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["tree", "xgboost"])
    args = parser.parse_args()

    processing_run = json.loads(LAB02_MANIFEST.read_text(encoding="utf-8"))
    session = boto3.Session(
        region_name=processing_run["region"],
        profile_name=os.getenv("AWS_PROFILE") or None,
    )
    job_name, image = train_model(session, processing_run, args.model)
    save_training_results(session, args.model, job_name, image, processing_run)


if __name__ == "__main__":
    main()
