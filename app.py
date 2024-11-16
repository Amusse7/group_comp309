from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: model.pkl not found")
    model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint to verify API is working"""
    return jsonify({'message': 'API is working!'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 503

    try:
        # Get data from POST request
        data = request.json
        
        # Print received data for debugging
        print("Received data:", data)
        
        # Convert data to format expected by model
        features = [
            float(data['feature1']), 
            float(data['feature2']), 
            float(data['feature3'])
        ]
        
        # Make prediction
        prediction = model.predict([features])
        
        # Return prediction
        return jsonify({
            'status': 'success',
            'prediction': int(prediction[0]),
            'received_data': data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Changed port to 5001
