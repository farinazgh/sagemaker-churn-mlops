"""Verify installed package versions and SDK v3 training/serving imports."""

from importlib.metadata import version

from sagemaker.train import ModelTrainer
from sagemaker.serve import ModelBuilder

for package in (
    "boto3",
    "sagemaker",
    "sagemaker-core",
    "sagemaker-train",
    "sagemaker-serve",
):
    print(f"{package}: {version(package)}")
assert version("sagemaker").split(".")[0] == "3", "Expected SageMaker SDK v3"
print(f"Imports PASS: {ModelTrainer.__name__}, {ModelBuilder.__name__}")
