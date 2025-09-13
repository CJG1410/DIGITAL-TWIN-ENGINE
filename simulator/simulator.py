import sqlite3
import numpy as np
import time
import json
import os
from datetime import datetime
import random

# --- Configuration Parameters ---
DB_NAME = "engine_data.db"
ANOMALY_LOG_PATH = "anomaly_log.json"
MAX_CYCLES = 150000 
DATA_UPDATE_INTERVAL = 0.01 
ANOMALY_PROBABILITY = 0.03 # Increased to 3% for richer anomaly data
ENVIRONMENT_CHANGE_INTERVAL = 1500 # Cycles after which the environment might change

# A list of sensor names for our simulation
SENSOR_NAMES = [
    'sensor_temperature', 'sensor_rpm', 'sensor_vibration',
    'sensor_fuel_pressure', 'sensor_ambient_temp', 'sensor_humidity'
]

# --- Sensor Data Generation Function ---
def generate_engine_state(cycle, max_cycles, environment_status):
    """
    Generates a realistic set of sensor readings.
    """
    base_readings = {
        'sensor_temperature': 917.5, 'sensor_rpm': 11996, 'sensor_vibration': 0.59,
        'sensor_fuel_pressure': 487, 'sensor_ambient_temp': 32, 'sensor_humidity': 18
    }
    
    # Calculate degradation based on remaining useful life (RUL)
    degradation_factor = (cycle / max_cycles) ** 1.5  # Non-linear degradation curve
    
    readings = {}
    for name, base_val in base_readings.items():
        # Apply degradation to core engine sensors
        if name in ['sensor_temperature', 'sensor_vibration', 'sensor_fuel_pressure']:
            noise = np.random.normal(0, 0.5 + degradation_factor * 2)
            reading = base_val * (1 + degradation_factor * 0.1) + noise
        elif name == 'sensor_rpm':
            noise = np.random.normal(0, 50)
            reading = base_val * (1 - degradation_factor * 0.05) + noise
        else:
            # No degradation for ambient sensors, but add noise
            noise = np.random.normal(0, 0.5)
            reading = base_val + noise

        # Adjust based on environment status
        if environment_status == "Extreme Weather":
            if name == 'sensor_ambient_temp': reading += random.uniform(5, 15)
            if name == 'sensor_humidity': reading += random.uniform(10, 20)
            if name in ['sensor_temperature', 'sensor_rpm']:
                reading *= random.uniform(1.01, 1.05)
        
        readings[name] = reading
    return readings

# --- Anomaly Injection Function ---
def inject_anomaly(data):
    """Randomly injects a significant spike in one of the core sensor readings."""
    data_with_anomaly = data.copy()
    is_anomaly = False
    if random.random() < ANOMALY_PROBABILITY:
        is_anomaly = True
        anomaly_type = random.choice(['vibration', 'temperature', 'rpm', 'fuel_pressure_drop', 'temp_spike_ambient'])
        if anomaly_type == 'vibration':
            data_with_anomaly['sensor_vibration'] *= random.uniform(5.0, 10.0)
            print(f"!!! ANOMALY INJECTED: VIBRATION at {datetime.now()} !!!")
        elif anomaly_type == 'temperature':
            data_with_anomaly['sensor_temperature'] *= random.uniform(1.5, 2.0)
            print(f"!!! ANOMALY INJECTED: TEMPERATURE at {datetime.now()} !!!")
        elif anomaly_type == 'rpm':
            data_with_anomaly['sensor_rpm'] *= random.uniform(0.6, 0.7)
            print(f"!!! ANOMALY INJECTED: RPM DROP at {datetime.now()} !!!")
        elif anomaly_type == 'fuel_pressure_drop':
            data_with_anomaly['sensor_fuel_pressure'] *= random.uniform(0.5, 0.7)
            print(f"!!! ANOMALY INJECTED: FUEL PRESSURE DROP at {datetime.now()} !!!")
        elif anomaly_type == 'temp_spike_ambient':
            data_with_anomaly['sensor_ambient_temp'] += random.uniform(20, 30)
            print(f"!!! ANOMALY INJECTED: AMBIENT TEMP SPIKE at {datetime.now()} !!!")
    return data_with_anomaly, is_anomaly

# --- Database Setup & Data Simulation ---
def setup_database():
    """Creates the database and sensor data table with a RUL column."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', DB_NAME)

    # Delete the old database file for a clean start
    if os.path.exists(db_path):
        os.remove(db_path)
        print("✅ Deleted old database file for a clean start.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    sensor_cols = ", ".join([f"{name} REAL" for name in SENSOR_NAMES])
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS engine_readings (
        timestamp TEXT PRIMARY KEY,
        rul INTEGER,
        engine_cycle INTEGER,
        {sensor_cols},
        environment_status TEXT
    );
    """
    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()

def simulate_data():
    """
    Simulates engine readings and inserts them into the database.
    """
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', DB_NAME)
    anomaly_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ANOMALY_LOG_PATH)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Initialize anomaly log file if it doesn't exist
    if not os.path.exists(anomaly_log_path):
        with open(anomaly_log_path, 'w') as f:
            json.dump([], f)

    print("🚀 Starting engine data simulation...")
    
    try:
        current_environment_status = "Normal"
        for cycle in range(1, MAX_CYCLES + 1):
            # Randomly change environment status
            if cycle % ENVIRONMENT_CHANGE_INTERVAL == 0:
                current_environment_status = random.choice(["Normal", "Extreme Weather", "High Altitude"])
                print(f"Environment status changed to: {current_environment_status}")

            rul = MAX_CYCLES - cycle
            readings = generate_engine_state(cycle, MAX_CYCLES, current_environment_status)
            data_point, is_anomaly = inject_anomaly(readings)
            
            # Prepare data for insertion
            timestamp = datetime.now().isoformat()
            
            # Prepare the row tuple
            row_data = (timestamp, rul, cycle) + tuple(data_point[s] for s in SENSOR_NAMES) + (current_environment_status,)
            
            insert_sql = f"INSERT INTO engine_readings VALUES ({', '.join(['?'] * len(row_data))});"
            cursor.execute(insert_sql, row_data)
            conn.commit()
            
            # Log anomaly if one occurred
            if is_anomaly:
                anomaly_log_entry = {
                    "timestamp": timestamp,
                    "cycle": cycle,
                    "data_point": data_point
                }
                with open(anomaly_log_path, 'r+') as f:
                    log_data = json.load(f)
                    log_data.append(anomaly_log_entry)
                    f.seek(0)
                    json.dump(log_data, f, indent=4)
            
            print(f"✅ Generated data for cycle {cycle}/{MAX_CYCLES} | RUL: {rul} cycles | Anomaly: {is_anomaly}")
            time.sleep(DATA_UPDATE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n🏁 Simulation stopped by user.")
    
    finally:
        print(f"🏁 Simulation complete. Data stored in `{DB_NAME}`")
        conn.close()

if __name__ == "__main__":
    setup_database()
    simulate_data()
