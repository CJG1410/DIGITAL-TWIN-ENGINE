import pandas as pd
import sqlite3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score
import shap
import json
import os
import numpy as np

# --- Configuration & File Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'engine_data.db')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.json')

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """
    Loads data from the SQLite DB, cleans it, and prepares it for modeling.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM engine_readings", conn)
        conn.close()
        
        # Drop irrelevant columns for modeling
        df.drop(['timestamp', 'environment_status'], axis=1, inplace=True)
        
        return df
    except Exception as e:
        print(f"❌ Error loading data from database: {e}")
        return None

# --- Model Training Functions ---
def train_rul_model(df):
    """
    Trains a RandomForestRegressor model to predict Remaining Useful Life (RUL).
    """
    print("Training RUL Prediction Model (Random Forest)...")
    
    # We use all sensor data and the engine cycle as features to predict RUL
    features = df.drop('rul', axis=1).columns.tolist()
    target = 'rul'
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # RandomForestRegressor for RUL prediction
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RUL Model Performance on Test Set:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")
    
    # Save the trained model
    joblib.dump(model, os.path.join(MODEL_DIR, 'rul_model.pkl'))
    
    # Save the list of feature names to a JSON file for the backend
    with open(FEATURES_PATH, 'w') as f:
        json.dump(features, f)
        
    print("✅ RUL model and feature names saved.")

def train_anomaly_model(df):
    """
    Trains an IsolationForest model for anomaly detection.
    """
    print("\nTraining Anomaly Detection Model (Isolation Forest)...")
    # Features for anomaly detection do not include the RUL or engine cycle
    features = [col for col in df.columns if 'sensor' in col]
    
    X = df[features]
    
    # IsolationForest for anomaly detection
    model = IsolationForest(contamination=0.03, random_state=42)
    model.fit(X)
    
    # Save the trained model
    joblib.dump(model, os.path.join(MODEL_DIR, 'anomaly_model.pkl'))
    print("✅ Anomaly Detection model saved as anomaly_model.pkl")

# --- Main Execution ---
if __name__ == "__main__":
    df = load_and_preprocess_data()
    if df is not None:
        if 'rul' not in df.columns or 'engine_cycle' not in df.columns:
            print("❌ Error: 'rul' or 'engine_cycle' column not found. Please check your data.")
        else:
            train_rul_model(df)
            train_anomaly_model(df)
