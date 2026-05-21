import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="c1ce131d-0f2f-4795-9c8a-cda41c55fa09",
    resource_group_name="ml-project-rg",
    workspace_name="karan-ml-workspace"
)

# Read GitHub commit information
github_sha = os.getenv("GITHUB_SHA", "local-run")

github_run_id = os.getenv("GITHUB_RUN_ID", "manual-run")


# Create model object
model = Model(
    path="model.joblib",
    name="iris-classification-model",
    type="custom_model",

    description = "Iris classification model trained via github",

    tags = {
        "framework" : "scikit-learn",
        "dataset" : "iris-dataset",
        "training type" : "automated",
        "pipeline" : "github-actions",
        "git_commit": github_sha,
        "github_run": github_run_id
    }
)

# Register model
registered_model = ml_client.models.create_or_update(model)

print("Model registered successfully")
print(registered_model.name)
print(registered_model.version)
print("Model tags:")
print(registered_model.tags)