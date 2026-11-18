"""Run and monitor a SageMaker Processing job."""

import argparse
import json
import os
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
MANIFEST = ART / "latest_run.json"
SAMPLE_KEY = "datasets/tabular/synthetic/churn.txt"
PROCESSING_DIR = "/opt/ml/processing"
SPLITS = ["train", "validation", "test"]
TAG = {"Key": "Project", "Value": "sagemaker-churn-mlops"}


def create_validated_aws_session():
    """1. Create an AWS session and validate the account and project resources."""
    load_dotenv(ROOT / ".env")
    region = os.getenv("AWS_REGION", "us-east-1")
    session = boto3.Session(
        region_name=region, profile_name=os.getenv("AWS_PROFILE") or None
    )
    account = get_validated_aws_account(session)
    bucket = f"sagemaker-churn-{account}-{region}"
    role = get_execution_role(session, region)
    bucket_tags = get_project_bucket_tags(session, bucket, account)
    validate_project_resources(bucket_tags, role)
    return session, account, bucket, role["Arn"]


def get_validated_aws_account(session):
    """2. Fetch the active AWS identity and validate its account and principal."""
    sts_client = session.client("sts")
    # 0. STS GetCallerIdentity checks which AWS identity and account this session uses.
    identity = sts_client.get_caller_identity()
    account = identity["Account"]
    if account != os.getenv("EXPECTED_AWS_ACCOUNT_ID", "").strip():
        raise ValueError("AWS account mismatch. Check .env.")
    if not identity["Arn"].startswith("arn:aws:") or identity["Arn"].endswith(":root"):
        raise ValueError("Use a non-root identity in the AWS commercial partition.")
    return account


def get_execution_role(session, region):
    """3. Retrieve the SageMaker execution role for the configured region."""
    iam_client = session.client("iam")
    # 1. IAM GetRole retrieves the execution role and its tags for validation.
    return iam_client.get_role(RoleName=f"SageMakerChurnLabExecutionRole-{region}")[
        "Role"
    ]


def get_project_bucket_tags(session, bucket, account):
    """4. Retrieve project bucket tags while enforcing the expected owner."""
    s3_client = session.client("s3")
    # 2. S3 GetBucketTagging verifies the expected project bucket and its ownership.
    return s3_client.get_bucket_tagging(Bucket=bucket, ExpectedBucketOwner=account)[
        "TagSet"
    ]


def validate_project_resources(bucket_tags, role):
    """5. Require the project tag on both the S3 bucket and execution role."""
    if TAG not in bucket_tags or TAG not in role.get("Tags", []):
        raise ValueError("Expected Lab 01 project resources")


def download_dataset_if_missing(session):
    """6. Download and validate the churn dataset when it is not already local."""
    if RAW.exists():
        print(f"Using {RAW}")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    s3_client = session.client("s3")
    temporary = RAW.with_suffix(".download")
    try:
        download_dataset_from_s3(s3_client, session.region_name, temporary)
        validate_and_install_dataset(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def download_dataset_from_s3(s3_client, region, destination):
    """7. Download the dataset from the primary sample bucket or its fallback."""
    try:
        # 3. S3 DownloadFile fetches the dataset from the primary sample bucket.
        s3_client.download_file(
            f"sagemaker-example-files-prod-{region}",
            SAMPLE_KEY,
            str(destination),
        )
    except ClientError:
        # 4. If the primary bucket does not have the object, fetch it from the fallback bucket.
        s3_client.download_file("sagemaker-sample-files", SAMPLE_KEY, str(destination))


def validate_and_install_dataset(temporary):
    """8. Validate the downloaded dataset and move it into the local data directory."""
    load_data(temporary)
    temporary.replace(RAW)


def print_dataset_summary():
    """9. Print the dataset's row count, column count, and churn rate."""
    df = load_data(RAW)
    print(
        f"Rows: {len(df)}, columns: {len(df.columns)}, churn rate: {df['Churn?'].mean():.4f}"
    )


def upload_inputs_and_submit_processing_job(session, account, bucket, role):
    """10. Upload processing inputs, start a SageMaker job, and save its manifest."""
    instance = os.getenv("SAGEMAKER_PROCESSING_INSTANCE_TYPE", "ml.m5.large")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"churn-p02-{timestamp}-{uuid4().hex[:8]}"
    prefix = f"lab02/runs/{name}"
    s3_client = session.client("s3")
    sagemaker_client = session.client("sagemaker")
    image = resolve_scikit_learn_image(session, instance)
    inputs, input_objects = upload_processing_inputs(s3_client, account, bucket, prefix)
    outputs = build_processing_outputs(bucket, prefix)
    submit_sagemaker_processing_job(
        sagemaker_client, name, role, image, instance, inputs, outputs
    )
    manifest = build_run_manifest(
        name, session.region_name, account, bucket, prefix, image, input_objects
    )
    save_run_manifest(manifest)
    print(f"Submitted: {name}")
    print(f"Output: s3://{bucket}/{prefix}/output/")
    return manifest


def resolve_scikit_learn_image(session, instance):
    """11. Resolve the SageMaker scikit-learn processing container image."""
    from sagemaker.core import image_uris

    return image_uris.retrieve(
        framework="sklearn",
        region=session.region_name,
        version="1.2-1",
        py_version="py3",
        instance_type=instance,
    )


def upload_processing_inputs(s3_client, account, bucket, prefix):
    """12. Upload the dataset and script and return their processing input definitions."""
    inputs = []
    input_objects = {}
    for folder, file in [("data", RAW), ("code", SCRIPT)]:
        key = f"{prefix}/input/{folder}/{file.name}"
        uri = f"s3://{bucket}/{key}"
        # 5. S3 PutObject uploads each input file so the processing job can read it.
        upload = s3_client.put_object(
            Bucket=bucket, Key=key, Body=file.read_bytes(), ExpectedBucketOwner=account
        )
        input_objects[folder] = {
            "uri": uri,
            "sha256": sha256(file),
            "version_id": upload.get("VersionId"),
        }
        inputs.append(
            {
                "InputName": folder,
                "S3Input": {
                    "S3Uri": uri,
                    "LocalPath": f"{PROCESSING_DIR}/input/{folder}",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            }
        )
    return inputs, input_objects


def build_processing_outputs(bucket, prefix):
    """13. Build the S3 destinations and local paths for processing outputs."""
    return [
        {
            "OutputName": folder,
            "S3Output": {
                "S3Uri": f"s3://{bucket}/{prefix}/output/{folder}/",
                "LocalPath": f"{PROCESSING_DIR}/output/{folder}",
                "S3UploadMode": "EndOfJob",
            },
        }
        for folder in SPLITS + ["metadata"]
    ]


def submit_sagemaker_processing_job(
    sagemaker_client, name, role, image, instance, inputs, outputs
):
    """14. Create the configured SageMaker Processing job."""
    # The native API is explicit about compute, inputs, outputs and entry point.
    # 6. SageMaker CreateProcessingJob starts processing the inputs and writes the configured outputs.
    sagemaker_client.create_processing_job(
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


def build_run_manifest(name, region, account, bucket, prefix, image, input_objects):
    """15. Create the run metadata used to track and verify a Processing job."""
    return {
        "job_name": name,
        "region": region,
        "account": account,
        "bucket": bucket,
        "prefix": prefix,
        "image": image,
        "input_objects": input_objects,
    }


def save_run_manifest(manifest):
    """16. Persist the latest Processing job manifest locally."""
    ART.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def wait_for_processing_job(session, job_name):
    """17. Wait for a SageMaker Processing job and raise if it did not complete."""
    sagemaker_client = session.client("sagemaker")
    print(f"Waiting for {job_name}")
    # 7. The waiter repeatedly calls DescribeProcessingJob until the job completes or stops.
    sagemaker_client.get_waiter("processing_job_completed_or_stopped").wait(
        ProcessingJobName=job_name, WaiterConfig={"Delay": 30, "MaxAttempts": 60}
    )
    # 8. SageMaker DescribeProcessingJob retrieves the final status and failure details.
    result = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
    if result["ProcessingJobStatus"] != "Completed":
        raise RuntimeError(
            f"{result['ProcessingJobStatus']}: {result.get('FailureReason', '')}"
        )


def download_and_verify_outputs(session, account, manifest):
    """18. Download job artifacts and validate their contents against the report."""
    output_dir = ART / manifest["job_name"]
    s3_client = session.client("s3")
    download_processing_outputs(s3_client, account, manifest, output_dir)
    schema, report = load_output_metadata(output_dir)
    verify_raw_dataset_checksum(report, manifest)
    total_rows = verify_dataset_splits(output_dir, schema, report)
    if total_rows != report["rows"]:
        raise ValueError("Total row count mismatch")
    print(f"Verification passed. Artifacts: {output_dir}")


def download_processing_outputs(s3_client, account, manifest, output_dir):
    """19. Download all expected split and metadata artifacts from S3."""
    files = [f"{split}/{split}.csv" for split in SPLITS]
    files += ["metadata/feature_schema.json", "metadata/data_report.json"]
    for file in files:
        destination = output_dir / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 9. S3 DownloadFile retrieves each output artifact for local verification.
        s3_client.download_file(
            manifest["bucket"],
            f"{manifest['prefix']}/output/{file}",
            str(destination),
            ExtraArgs={"ExpectedBucketOwner": account},
        )


def load_output_metadata(output_dir):
    """20. Load the feature schema and data report generated by processing."""
    schema = json.loads(
        (output_dir / "metadata/feature_schema.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (output_dir / "metadata/data_report.json").read_text(encoding="utf-8")
    )
    return schema, report


def verify_raw_dataset_checksum(report, manifest):
    """21. Ensure the processing report matches the uploaded raw dataset."""
    if report["raw_sha256"] != manifest["input_objects"]["data"]["sha256"]:
        raise ValueError("Raw dataset checksum mismatch")


def verify_dataset_splits(output_dir, schema, report):
    """22. Validate each generated split and return the combined row count."""
    total_rows = 0
    for split in SPLITS:
        path = output_dir / split / f"{split}.csv"
        df = pd.read_csv(path, header=None)
        expected = report["splits"][split]
        if df.shape != (expected["rows"], len(schema["feature_columns"]) + 1):
            raise ValueError(f"Unexpected dimensions: {split}")
        if df.isna().any().any() or set(df[0].unique()) != {0, 1}:
            raise ValueError(f"Invalid data: {split}")
        if (
            sha256(path) != expected["sha256"]
            or abs(df[0].mean() - expected["churn_rate"]) > 1e-9
        ):
            raise ValueError(f"Checksum or churn rate mismatch: {split}")
        total_rows += len(df)
        print(f"{split}: {df.shape}, churn rate: {df[0].mean():.4f}")
    return total_rows


def main():
    """0. Run dataset preparation, processing, monitoring, and output verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Reuse latest job without launching compute",
    )
    args = parser.parse_args()
    session, account, bucket, role = create_validated_aws_session()

    if args.verify_existing:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if (manifest["account"], manifest["bucket"], manifest["region"]) != (
            account,
            bucket,
            session.region_name,
        ):
            raise ValueError("Latest run belongs to another AWS configuration")
    else:
        download_dataset_if_missing(session)
        print_dataset_summary()
        manifest = upload_inputs_and_submit_processing_job(
            session, account, bucket, role
        )

    wait_for_processing_job(session, manifest["job_name"])
    download_and_verify_outputs(session, account, manifest)


if __name__ == "__main__":
    main()
