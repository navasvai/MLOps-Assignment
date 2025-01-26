import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
def load_data():
    data = load_iris()
    return data.data, data.target

# Train the model
def train_model(X_train, y_train, max_depth):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

if __name__ == "__main__":
    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow experiment
    mlflow.set_experiment("Iris Experiment")
    for max_depth in [3, 5, 10]:  # Run experiments with different hyperparameters
        with mlflow.start_run():
            # Train model
            model = train_model(X_train, y_train, max_depth)

            # Evaluate model
            accuracy = evaluate_model(model, X_test, y_test)

            # Log parameters, metrics, and model
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "decision_tree_model")

            print(f"Experiment with max_depth={max_depth} logged. Accuracy: {accuracy:.2f}")
