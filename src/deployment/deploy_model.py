from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

# Connect to Azure ML workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="c1ce131d-0f2f-4795-9c8a-cda41c55fa09",
    resource_group_name="ml-project-rg",
    workspace_name="karan-ml-workspace"
)

# Get latest registered model
latest_model = ml_client.models.get(
    name="iris-classification-model",
    label="latest"
)

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue-deployment",

    endpoint_name="iris-ep-v10",

    model=latest_model,

    instance_type="Standard_DS3_v2",

    instance_count=1
)

# Create or update deployment
ml_client.online_deployments.begin_create_or_update(
    deployment
).result()

print("Deployment completed successfully")