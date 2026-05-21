from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="c1ce131d-0f2f-4795-9c8a-cda41c55fa09",
    resource_group_name="ml-project-rg",
    workspace_name="karan-ml-workspace"
)


# Create custom environment
custom_env = Environment(
    name="train-env",
    description="Training environment",
    conda_file="./environments/train-env.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)


# Create command job
job = command(
    code="./src",

    command="""
    python validate_data.py &&
    python train.py &&
    python register_model.py
    """,

    environment=custom_env,

    compute="cpu-cluster"
)


# Submit job
returned_job = ml_client.jobs.create_or_update(job)


# Print job name
print("Job submitted successfully")
print(returned_job.name)