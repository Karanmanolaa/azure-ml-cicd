from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
data = load_iris(as_frame=True)

# Create dataframe
df = data.frame

# Expected columns
required_columns = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target"
]

# Validate columns
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

# Fail if missing columns found
if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# Validate null values
if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Dataset contains null values"
    )

print("Data validation passed successfully")