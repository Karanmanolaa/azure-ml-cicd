import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Start MLflow run
mlflow.start_run()

# Load dataset
data = load_iris()

# Split features and labels
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Log metric
mlflow.log_metric("accuracy", accuracy)

# Print result
print(f"Accuracy: {accuracy}")

#minimum accuracy threshold 

# Minimum accuracy threshold
minimum_accuracy = 0.80

# Fail job if accuracy too low
if accuracy < minimum_accuracy:
    raise ValueError(
        f"Model accuracy below threshold: {accuracy}"
    )

# Save trained model
joblib.dump(model, "model.joblib")

print("Model saved successfully")

# End MLflow run
mlflow.end_run()