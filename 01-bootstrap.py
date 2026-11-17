"""Create or verify dedicated, tagged S3/IAM foundations; never launch compute."""

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
TAG = {"Key": "Project", "Value": "sagemaker-churn-mlops"}
POLICY_NAME = "ChurnLabExecutionPolicy"
TRUST = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def policy_doc(bucket, region, account):
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ProjectBucket",
                "Effect": "Allow",
                "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": f"arn:aws:s3:::{bucket}",
            },
            {
                "Sid": "ProjectObjects",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:PutObject",
                    "s3:AbortMultipartUpload",
                ],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
            {
                "Sid": "EcrAuth",
                "Effect": "Allow",
                "Action": "ecr:GetAuthorizationToken",
                "Resource": "*",
            },
            {
                "Sid": "EcrRead",
                "Effect": "Allow",
                "Action": [
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchCheckLayerAvailability",
                ],
                "Resource": f"arn:aws:ecr:{region}:*:repository/*",
            },
            {
                "Sid": "JobLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                "Resource": f"arn:aws:logs:{region}:{account}:log-group:/aws/sagemaker/*",
            },
            {
                "Sid": "DescribeLogGroups",
                "Effect": "Allow",
                "Action": "logs:DescribeLogGroups",
                "Resource": "*",
            },
            {
                "Sid": "Metrics",
                "Effect": "Allow",
                "Action": "cloudwatch:PutMetricData",
                "Resource": "*",
            },
        ],
    }


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--apply", action="store_true", help="Create and configure project resources"
    )
    modes.add_argument(
        "--verify",
        action="store_true",
        help="Inspect resources and perform a tiny S3 round trip",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    region = os.getenv("AWS_REGION", "us-east-1")
    profile = os.getenv("AWS_PROFILE") or None
    expected = os.getenv("EXPECTED_AWS_ACCOUNT_ID", "339712939282").strip()
    session = boto3.Session(profile_name=profile, region_name=region)
    identity = session.client("sts").get_caller_identity()
    account = identity["Account"]
    require(
        expected == account,
        "AWS account mismatch or EXPECTED_AWS_ACCOUNT_ID is unset. No resources changed.",
    )
    require(
        identity["Arn"].startswith("arn:aws:"),
        "This lab supports the standard AWS commercial partition only.",
    )
    require(not identity["Arn"].endswith(":root"), "Do not use root credentials.")
    bucket = f"sagemaker-churn-{account}-{region}"
    role_name = f"SageMakerChurnLabExecutionRole-{region}"
    s3 = session.client("s3")
    iam = session.client("iam")
    sm = session.client("sagemaker")
    expected_owner = {"ExpectedBucketOwner": account}
    print(f"Account: {account} | Region: {region}")
    print(f"Bucket: {bucket} | Role: {role_name}")

    if args.apply:
        try:
            s3.head_bucket(Bucket=bucket, **expected_owner)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            kwargs = {"Bucket": bucket}
            if region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
            s3.create_bucket(**kwargs)
            s3.put_bucket_tagging(
                Bucket=bucket, Tagging={"TagSet": [TAG]}, **expected_owner
            )
        else:
            tags = s3.get_bucket_tagging(Bucket=bucket, **expected_owner)["TagSet"]
            require(
                TAG in tags,
                "Existing bucket is not tagged for this lab. Refusing to modify it.",
            )
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
            **expected_owner,
        )
        s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            },
            **expected_owner,
        )
        s3.put_bucket_versioning(
            Bucket=bucket,
            VersioningConfiguration={"Status": "Enabled"},
            **expected_owner,
        )
        try:
            role = iam.get_role(RoleName=role_name)["Role"]
        except iam.exceptions.NoSuchEntityException:
            role = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(TRUST),
                Description="SageMaker churn lab job execution role",
                Tags=[TAG],
            )["Role"]
        else:
            require(
                TAG in role.get("Tags", []),
                "Existing role is not tagged for this lab. Refusing to modify it.",
            )
            require(
                role["AssumeRolePolicyDocument"] == TRUST,
                "Existing role trust differs; inspect it manually.",
            )
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=POLICY_NAME,
            PolicyDocument=json.dumps(policy_doc(bucket, region, account)),
        )
        print("Bootstrap applied. Run --verify next.")
    else:
        s3.head_bucket(Bucket=bucket, **expected_owner)
        require(
            TAG in s3.get_bucket_tagging(Bucket=bucket, **expected_owner)["TagSet"],
            "Bucket tag missing",
        )
        location = (
            s3.get_bucket_location(Bucket=bucket, **expected_owner)[
                "LocationConstraint"
            ]
            or "us-east-1"
        )
        require(location == region, f"Bucket region mismatch: {location}")
        config = s3.get_public_access_block(Bucket=bucket, **expected_owner)[
            "PublicAccessBlockConfiguration"
        ]
        require(
            all(
                config.get(k)
                for k in (
                    "BlockPublicAcls",
                    "IgnorePublicAcls",
                    "BlockPublicPolicy",
                    "RestrictPublicBuckets",
                )
            ),
            "Public access block incomplete",
        )
        rules = s3.get_bucket_encryption(Bucket=bucket, **expected_owner)[
            "ServerSideEncryptionConfiguration"
        ]["Rules"]
        require(
            any(
                r["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256"
                for r in rules
            ),
            "Expected SSE-S3 encryption",
        )
        require(
            s3.get_bucket_versioning(Bucket=bucket, **expected_owner).get("Status")
            == "Enabled",
            "Versioning not enabled",
        )
        role = iam.get_role(RoleName=role_name)["Role"]
        require(TAG in role.get("Tags", []), "Role tag missing")
        require(role["AssumeRolePolicyDocument"] == TRUST, "Role trust mismatch")
        actual_policy = iam.get_role_policy(RoleName=role_name, PolicyName=POLICY_NAME)[
            "PolicyDocument"
        ]
        require(
            actual_policy == policy_doc(bucket, region, account),
            "Inline execution policy mismatch",
        )
        key = f"lab01/smoke/{uuid4().hex}.txt"
        payload = b"SageMaker Lab 01 connectivity check\n"
        result = s3.put_object(Bucket=bucket, Key=key, Body=payload, **expected_owner)
        received = s3.get_object(Bucket=bucket, Key=key, **expected_owner)
        require(received["Body"].read() == payload, "S3 round-trip mismatch")
        print(f"S3 round trip PASS: s3://{bucket}/{key}")
        print(f"S3 object version: {result.get('VersionId', 'not yet available')}")
        sm.list_training_jobs(MaxResults=1)
        print("SageMaker control-plane API PASS")
        print(
            "IAM trust and inline policy PASS (not a test of actual service assumption)"
        )
        print("All Lab 01 checks passed.")
    print(f"SAGEMAKER_ROLE_ARN={role['Arn']}")
    print(f"S3_BUCKET={bucket}")


if __name__ == "__main__":
    main()
