import os
import pandas as pd
import mlflow
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle


def save_dataset():
    """
    Save the Iris dataset as a CSV file locally.
    """
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/iris.csv", index=False)
    print("Iris dataset saved to data/iris.csv")


def load_data():
    """
    Load the dataset from a local CSV file.

    Returns:
        X (array): Feature matrix.
        y (array): Target vector.
    """
    data = pd.read_csv("data/iris.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Args:
        X_train (array): Training features.
        y_train (array): Training labels.

    Returns:
        best_model (DecisionTreeClassifier): The best model found.
        best_params (dict): Best hyperparameters.
    """
    param_grid = {
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best Hyperparameters:", best_params)
    return best_model, best_params


if __name__ == "__main__":
    # Save dataset locally if not already saved
    if not os.path.exists("data/iris.csv"):
        save_dataset()

    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Perform hyperparameter tuning
    best_model, best_params = hyperparameter_tuning(X_train, y_train)

    # Evaluate the best model
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Best Model Accuracy: {accuracy:.2f}")

    # Save the best model
    with open("best_model.pkl", "wb") as file:
        pickle.dump(best_model, file)
    print("Best model saved to best_model.pkl")
