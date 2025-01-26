import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset
def load_data():
    """
    Load the Iris dataset.
    Returns:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
    """
    data = load_iris()
    return data.data, data.target


# Train the model
def train_model(X_train, y_train):
    """
    Train a Decision Tree Classifier.
    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
    Returns:
        model (DecisionTreeClassifier): Trained model.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy score.
    Args:
        model (DecisionTreeClassifier): Trained model.
        X_test (ndarray): Test features.
        y_test (ndarray): Test labels.
    Returns:
        accuracy (float): Model accuracy score.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


if __name__ == "__main__":
    # Load data
    X, y = load_data()

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the model
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)
    print("Model saved successfully to model.pkl")
