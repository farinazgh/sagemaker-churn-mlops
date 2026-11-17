"""Run and monitor a SageMaker Processing job."""

import argparse
import json
import os
import time

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import boto3
import pandas as pd

from botocore.exceptions import ClientError
from dotenv import load_dotenv
from preprocess import load_data, sha256

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data/raw/churn.txt"
SCRIPT = ROOT / "preprocess.py"
ART = ROOT / "artifacts/lab02"

TAG = {"Key": "Project", "Value": "sagemaker-churn-mlops"}
KEY = "datasets/tabular/synthetic/churn.txt"

PROCESSING_DIR = "/opt/ml/processing"
SPLITS = ["train", "validation", "test"]


# Connect to AWS and validate resources
def context():
    load_dotenv(ROOT / ".env")

    region = os.getenv("AWS_REGION", "us-east-1")
    profile = os.getenv("AWS_PROFILE") or None
    expected_account = os.getenv("EXPECTED_AWS_ACCOUNT_ID", "").strip()

    if not expected_account:
        raise RuntimeError("Set EXPECTED_AWS_ACCOUNT_ID in the project .env file.")

    session = boto3.Session(profile_name=profile, region_name=region)

    sts = session.client("sts")
    identity = sts.get_caller_identity()

    account = identity["Account"]
    arn = identity["Arn"]

    if account != expected_account:
        raise RuntimeError("AWS account does not match EXPECTED_AWS_ACCOUNT_ID.")

    if not arn.startswith("arn:aws:") or arn.endswith(":root"):
        raise RuntimeError("Unsupported AWS identity.")

    bucket = f"sagemaker-churn-{account}-{region}"

    s3 = session.client("s3")

    bucket_tags = s3.get_bucket_tagging(Bucket=bucket, ExpectedBucketOwner=account)[
        "TagSet"
    ]

    if TAG not in bucket_tags:
        raise RuntimeError("Invalid project bucket.")

    iam = session.client("iam")

    role = iam.get_role(RoleName=f"SageMakerChurnLabExecutionRole-{region}")["Role"]

    if TAG not in role.get("Tags", []):
        raise RuntimeError("Invalid project role.")

    return session, account, bucket, role["Arn"]


# Download the dataset
def fetch(session):
    print(f"Fetching dataset: {RAW}")
    if RAW.exists():
        print(f"Using existing dataset: {RAW}")
        load_data(RAW)
        return

    RAW.parent.mkdir(parents=True, exist_ok=True)

    s3 = session.client("s3")
    temp_file = RAW.with_suffix(".download")

    buckets = [
        f"sagemaker-example-files-prod-{session.region_name}",
        "sagemaker-sample-files",
    ]

    errors = []

    for bucket in buckets:
        try:
            s3.download_file(bucket, KEY, str(temp_file))

            load_data(temp_file)
            temp_file.replace(RAW)

            print(f"Downloaded dataset from {bucket}")
            return

        except (ClientError, ValueError) as error:
            errors.append(f"{bucket}: {error}")
            temp_file.unlink(missing_ok=True)

    raise RuntimeError(f"Dataset download failed: {errors}. Copy churn.txt to {RAW}.")


# Inspect the dataset
def inspect():
    print("inspecting dataset:", RAW)
    df = load_data(RAW)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Churn rate: {df['Churn?'].mean():.4f}")

    print(df.describe().to_string())


# Submit a SageMaker Processing job
def submit(session, account, bucket, role, instance):
    from sagemaker.core import image_uris

    load_data(RAW)

    sm = session.client("sagemaker")
    s3 = session.client("s3")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = uuid4().hex[:8]

    name = f"churn-p02-{timestamp}-{run_id}"
    prefix = f"lab02/runs/{name}"

    image = image_uris.retrieve(
        framework="sklearn",
        region=session.region_name,
        version="1.2-1",
        py_version="py3",
        instance_type=instance,
    )

    manifest = {
        "job_name": name,
        "region": session.region_name,
        "account": account,
        "bucket": bucket,
        "prefix": prefix,
        "image": image,
        "input_objects": {},
    }

    # Upload data and preprocessing code

    files = {"data": RAW, "code": SCRIPT}

    inputs = []

    for kind, file in files.items():
        key = f"{prefix}/input/{kind}/{file.name}"
        uri = f"s3://{bucket}/{key}"

        response = s3.put_object(
            Bucket=bucket, Key=key, Body=file.read_bytes(), ExpectedBucketOwner=account
        )

        manifest["input_objects"][kind] = {
            "uri": uri,
            "version_id": response.get("VersionId"),
            "sha256": sha256(file),
        }

        inputs.append(
            {
                "InputName": kind,
                "S3Input": {
                    "S3Uri": uri,
                    "LocalPath": f"{PROCESSING_DIR}/input/{kind}",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            }
        )

    # Configure job outputs

    outputs = []

    for output_name in SPLITS + ["metadata"]:
        outputs.append(
            {
                "OutputName": output_name,
                "S3Output": {
                    "S3Uri": f"s3://{bucket}/{prefix}/output/{output_name}/",
                    "LocalPath": f"{PROCESSING_DIR}/output/{output_name}",
                    "S3UploadMode": "EndOfJob",
                },
            }
        )

    # Save the run manifest

    ART.mkdir(parents=True, exist_ok=True)

    manifest_path = ART / "latest_run.json"

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Create the managed Processing job

    sm.create_processing_job(
        ProcessingJobName=name,
        RoleArn=role,
        AppSpecification={
            "ImageUri": image,
            "ContainerEntrypoint": [
                "python3",
                f"{PROCESSING_DIR}/input/code/preprocess.py",
            ],
            "ContainerArguments": [
                "--input-file",
                f"{PROCESSING_DIR}/input/data/churn.txt",
                "--output-dir",
                f"{PROCESSING_DIR}/output",
            ],
        },
        ProcessingInputs=inputs,
        ProcessingOutputConfig={"Outputs": outputs},
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": instance,
                "VolumeSizeInGB": 10,
            }
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1200},
        NetworkConfig={"EnableNetworkIsolation": True},
        Tags=[TAG],
    )

    print(f"Submitted: {name}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: s3://{bucket}/{prefix}/output/")


def get_manifest(account, bucket, region):
    path = ART / "latest_run.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    if (
        manifest["account"] != account
        or manifest["bucket"] != bucket
        or manifest["region"] != region
    ):
        raise RuntimeError("Run configuration mismatch.")

    return manifest


# Check job status
def status(sm, name):
    response = sm.describe_processing_job(ProcessingJobName=name)

    job_status = response["ProcessingJobStatus"]

    print(job_status, response.get("FailureReason", ""))

    return job_status


# Wait for job completion
def wait(sm, name):
    while True:
        job_status = status(sm, name)

        if job_status == "Completed":
            return

        if job_status in ["Failed", "Stopped"]:
            raise SystemExit(1)

        time.sleep(30)


# Verify the Processing job outputs
def verify(session, account, bucket, manifest):
    sm = session.client("sagemaker")
    s3 = session.client("s3")

    name = manifest["job_name"]
    prefix = manifest["prefix"]

    if status(sm, name) != "Completed":
        raise RuntimeError("Job has not completed successfully.")

    output_dir = ART / name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "train/train.csv",
        "validation/validation.csv",
        "test/test.csv",
        "metadata/feature_schema.json",
        "metadata/data_report.json",
    ]

    # Download output files

    for file in files:
        destination = output_dir / file

        destination.parent.mkdir(parents=True, exist_ok=True)

        s3.download_file(
            bucket,
            f"{prefix}/output/{file}",
            str(destination),
            ExtraArgs={"ExpectedBucketOwner": account},
        )

    # Read metadata

    schema = json.loads((output_dir / "metadata/feature_schema.json").read_text())

    report = json.loads((output_dir / "metadata/data_report.json").read_text())

    # Verify the original dataset checksum

    original_checksum = manifest["input_objects"]["data"]["sha256"]

    assert report["raw_sha256"] == original_checksum

    # Verify total number of rows

    total_rows = 0

    for split in report["splits"].values():
        total_rows += split["rows"]

    assert total_rows == report["rows"]

    # Verify each dataset

    for split in SPLITS:
        file = output_dir / split / f"{split}.csv"

        df = pd.read_csv(file, header=None)

        split_report = report["splits"][split]

        expected_rows = split_report["rows"]
        expected_columns = 1 + len(schema["feature_columns"])

        churn_rate = df[0].mean()

        assert sha256(file) == split_report["sha256"]
        assert len(df) == expected_rows
        assert len(df.columns) == expected_columns
        assert set(df[0].unique()) == {0, 1}
        assert not df.isna().any().any()
        assert abs(churn_rate - split_report["churn_rate"]) < 1e-9

        print(f"{split}: {df.shape}, churn_rate={churn_rate:.4f}, checksum PASS")

    print("All output checks PASS.")
    print(f"Local artifacts: {output_dir}")


# Stop a running job
def stop(sm, name):
    if status(sm, name) == "InProgress":
        sm.stop_processing_job(ProcessingJobName=name)

        print("Stop requested; check status until Stopped.")
    else:
        print("No running job to stop.")


# Main program
def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "action",
        choices=["fetch", "inspect", "submit", "status", "wait", "verify", "stop"],
    )

    parser.add_argument(
        "--instance-type",
        choices=["ml.m5.large", "ml.m5.xlarge"],
        default="ml.m5.large",
    )

    args = parser.parse_args()

    if args.action == "inspect":
        inspect()
        return

    session, account, bucket, role = context()

    if args.action == "fetch":
        fetch(session)
        return

    if args.action == "submit":
        submit(session, account, bucket, role, args.instance_type)
        return

    sm = session.client("sagemaker")

    manifest = get_manifest(account, bucket, session.region_name)

    name = manifest["job_name"]

    if args.action == "status":
        status(sm, name)
    elif args.action == "wait":
        wait(sm, name)
    elif args.action == "verify":
        verify(session, account, bucket, manifest)
    elif args.action == "stop":
        stop(sm, name)


if __name__ == "__main__":
    main()
