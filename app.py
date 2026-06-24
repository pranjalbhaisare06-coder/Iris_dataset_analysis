from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('iris_knn_model.pkl')

# Map numerical predictions back to species names
species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

@app.route('/')
def home():
    return "<h1>Iris Species Prediction API</h1><p>Send a POST request to /predict with features to get a prediction.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Ensure all required features are present and in correct order
        required_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        input_features = [data[feature] for feature in required_features]
        
        # Convert to DataFrame, ensuring column names match training data
        input_df = pd.DataFrame([input_features], columns=required_features)
        
        prediction = model.predict(input_df)
        predicted_species = species_map[prediction[0]]
        
        return jsonify({"prediction": predicted_species})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run()
