import os
import sqlite3
import pandas as pd
import joblib
import numpy as np
import json
import shap
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Configuration & File Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_NAME = 'engine_data.db'
DB_PATH = os.path.join(BASE_DIR, DATABASE_NAME)
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RUL_MODEL_PATH = os.path.join(MODELS_DIR, 'rul_model.pkl')
ANOMALY_MODEL_PATH = os.path.join(MODELS_DIR, 'anomaly_model.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'features.json')

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enables Cross-Origin Resource Sharing for our frontend

# --- Load Models and Features (Global Scope) ---
try:
    rul_model = joblib.load(RUL_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    
    # Load the feature names used for training
    with open(FEATURES_PATH, 'r') as f:
        features = json.load(f)
        
    print("✅ Successfully loaded RUL and Anomaly Detection models.")
    
    # Create a SHAP Explainer for the RUL model
    # We use a TreeExplainer for tree-based models like RandomForest
    explainer = shap.TreeExplainer(rul_model)
    print("✅ SHAP Explainer for RUL model created.")
    
except FileNotFoundError:
    print("❌ Error: Model files not found. Please ensure `train_models.py` has been run.")
    rul_model = None
    anomaly_model = None
    explainer = None
    features = None
    
# --- Database Connection Helper ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- API Endpoints ---
@app.route('/')
def home():
    """A simple homepage to confirm the API is running."""
    return "🚀 Digital Twin Backend API is running!"

@app.route('/latest', methods=['GET'])
def get_latest_data():
    """Returns the latest sensor data from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM engine_readings ORDER BY timestamp DESC LIMIT 1;")
    latest_row = cursor.fetchone()
    conn.close()
    
    if latest_row:
        latest_data = dict(latest_row)
        return jsonify(latest_data)
    else:
        return jsonify({"error": "No data found in the database."}), 404

@app.route('/history', methods=['GET'])
def get_all_history():
    """Returns all sensor data history from the database."""
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM engine_readings ORDER BY timestamp ASC", conn)
    conn.close()
    
    history = df.to_dict('records')
    return jsonify(history)

@app.route('/predict/all', methods=['POST'])
def predict_all():
    """
    Predicts RUL, detects anomalies, and provides SHAP explanations in a single request.
    This is more efficient for the real-time digital twin.
    """
    if not rul_model or not anomaly_model or not explainer:
        return jsonify({"error": "Models or Explainer not loaded. Check server logs."}), 500
        
    try:
        data = request.get_json(force=True)
        
        # Prepare data for RUL prediction (ensure the order of features is correct)
        # Note: We are not using `environment_status` as a feature for RUL prediction
        rul_features_dict = {k: data[k] for k in features if k in data}
        
        # Ensure 'rul' is dropped and 'engine_cycle' is included if present
        if 'rul' in rul_features_dict:
            del rul_features_dict['rul']
        
        rul_input_data = pd.DataFrame([rul_features_dict])
        
        # Predict RUL
        rul_prediction = rul_model.predict(rul_input_data)[0]
        
        # Prepare data for Anomaly detection
        anomaly_features = [col for col in rul_input_data.columns if 'sensor' in col]
        anomaly_input_data = np.array([rul_input_data[name].iloc[0] for name in anomaly_features]).reshape(1, -1)
        
        # Predict Anomaly
        anomaly_prediction = anomaly_model.predict(anomaly_input_data)[0]
        is_anomaly = True if anomaly_prediction == -1 else False

        # Calculate SHAP values for the RUL prediction
        shap_values_obj = explainer.shap_values(rul_input_data)
        
        # Create a dictionary for SHAP values
        shap_dict = {
            feature: float(shap_values_obj[0][i]) for i, feature in enumerate(rul_input_data.columns)
        }
        
        return jsonify({
            "rul_prediction": max(0, int(rul_prediction)),
            "is_anomaly": is_anomaly,
            "shap_explanation": shap_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible to other machines on the network
    # This is important for the Unity app which will run on a different process
    app.run(debug=True, host='0.0.0.0', port=5000)
