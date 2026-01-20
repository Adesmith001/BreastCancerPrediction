"""
Breast Cancer Prediction System
CSC415 Holiday Assignment - Project 5
Author: SOMADE TOLUWANI (22CH032062)
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'breast_cancer_model.pkl')
model_data = None

def get_model():
    global model_data
    if model_data is None:
        model_data = joblib.load(MODEL_PATH)
    return model_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        md = get_model()
        
        # Prepare features in the exact order: radius, texture, area, smoothness, concavity
        features = np.array([[
            float(data['radius_mean']),
            float(data['texture_mean']),
            float(data['area_mean']),
            float(data['smoothness_mean']),
            float(data['concavity_mean'])
        ]])
        
        # Scale
        features_scaled = md['scaler'].transform(features)
        
        # Predict
        prediction = md['model'].predict(features_scaled)[0]
        # Map 0 -> Malignant, 1 -> Benign (from dataset)
        result = "Benign" if prediction == 1 else "Malignant"
        
        # Probability
        probs = md['model'].predict_proba(features_scaled)[0]
        confidence = float(max(probs))
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': confidence,
            'is_benign': bool(prediction == 1)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
