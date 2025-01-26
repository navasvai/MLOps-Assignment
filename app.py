import pickle
import pandas as pd
from flask import Flask, request, jsonify


# Load the saved model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)


app = Flask(__name__)


@app.route("/")
def home():
    """
    Home endpoint to indicate the API is running.
    """
    return "Model API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Make predictions using the trained model.
    """
    data = request.get_json()  # Input should be a JSON object
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify({"predictions": predictions.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
