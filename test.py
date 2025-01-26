import unittest
import pickle
import os
from train import save_dataset, load_data, train_model, evaluate_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class TestMLProject(unittest.TestCase):
    def setUp(self):
        """
        Set up the dataset for testing.
        """
        if not os.path.exists("data/iris.csv"):
            save_dataset()
        self.X, self.y = load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def test_data_loading(self):
        """
        Test that data loads correctly.
        """
        self.assertEqual(
            len(self.X), len(self.y), "Number of features and labels mismatch"
        )

    def test_model_training(self):
        """
        Test that the model trains without errors.
        """
        model = train_model(self.X_train, self.y_train, max_depth=5)
        self.assertIsInstance(
            model, DecisionTreeClassifier, "Model is not a DecisionTreeClassifier"
        )

    def test_model_evaluation(self):
        """
        Test that the evaluation returns a valid accuracy score.
        """
        model = train_model(self.X_train, self.y_train, max_depth=5)
        accuracy = evaluate_model(model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0, "Accuracy is less than 0")
        self.assertLessEqual(accuracy, 1, "Accuracy is greater than 1")

    def test_model_saving(self):
        """
        Test that the model is saved to a file and can be loaded.
        """
        model = train_model(self.X_train, self.y_train, max_depth=5)
        with open("test_model.pkl", "wb") as file:
            pickle.dump(model, file)

        # Check that the file exists and can be loaded
        with open("test_model.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        self.assertIsInstance(
            loaded_model,
            DecisionTreeClassifier,
            "Saved model is not a DecisionTreeClassifier",
        )


if __name__ == "__main__":
    unittest.main()
