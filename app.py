# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Import custom physics-informed classes to allow pickle to safely deserialize them
try:
    import train_model
    # --- Bulletproof Pickle Namespace Alignment ---
    # Force Python's __main__ namespace to recognize our custom classes before unpickling
    sys.modules['__main__'].PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    sys.modules['__main__'].calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
except ImportError:
    # Fallback in case train_model.py hasn't been parsed yet during initial launch
    PhysicsInformedEstimator = None
    calculate_physics_power_vectorized = None

app = Flask(__name__)

# System Memory Cache Arrays (State Ledger)
SYSTEM_STATE = {
    "engine_type": 0, "speed": 120.0, "weight": 450.0, "gradient": 0.5, "distance": 50.0,
    "passengers": 320, "temp": 24.0, "aux_load": 45.0, "headwind": 15.0, "drag_coeff": 0.28,
    "rolling_res": 0.0015, "adhesion": 0.32, "inverter_eff": 94.0, "gear_ratio": 4.12,
    "wheel_diam": 920.0, "motor_freq": 60.0, "brake_pressure": 420.0, "regen": 0.28,
    "control_override": 1, "simulation_pass": 1,
    "pred_kwh_per_hour": 182.4, "pred_total_kwh": 76.0,
    "pred_liters_per_hour": 0.0, "pred_total_liters": 0.0
}

LEDGER_HISTORY = []

# Core Estimator Loading Verification Routines
MODELS = None
def init_inference_engine():
    global MODELS
    model_path = 'rail_ai_models.pkl'
    if not os.path.exists(model_path):
        import train_model
        train_model.train_system_estimators()
        
    # Re-import to guarantee namespace mapping is registered in sys.modules
    global PhysicsInformedEstimator, calculate_physics_power_vectorized
    import train_model
    sys.modules['__main__'].PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    sys.modules['__main__'].calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    with open(model_path, 'rb') as f:
        MODELS = pickle.load(f)

init_inference_engine()

# --- Page Mappings ---
@app.route('/')
def route_dashboard(): return render_template('index.html')

@app.route('/predict_page')
def route_predict_page(): return render_template('predict.html')

@app.route('/analytics')
def route_analytics(): return render_template('analytics.html')

@app.route('/charts')
def route_charts(): return render_template('charts.html')

@app.route('/history')
def route_history(): return render_template('history.html')


# --- Real-Time Operational API Framework ---
@app.route('/api/get-system-state', methods=['GET'])
def get_system_state():
    return jsonify(SYSTEM_STATE)

@app.route('/api/get-history-ledger', methods=['GET'])
def get_history_ledger():
    return jsonify(LEDGER_HISTORY)

@app.route('/api/clear-history-ledger', methods=['POST'])
def clear_history_ledger():
    global LEDGER_HISTORY
    LEDGER_HISTORY = []
    return jsonify({"success": True})

@app.route('/predict', methods=['POST'])
def execute_inference():
    global SYSTEM_STATE
    try:
        req = request.json
        
        # Parse inputs ordered sequentially by feature matrix layout
        feature_cols = [
            'engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp',
            'aux_load', 'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff',
            'gear_ratio', 'wheel_diam', 'motor_freq', 'brake_pressure', 'regen',
            'control_override', 'simulation_pass'
        ]
        
        input_data = [float(req.get(col, SYSTEM_STATE.get(col))) for col in feature_cols]
        feature_vector = np.array([input_data])
        
        engine_type = int(req.get('engine_type', 0))
        
        # Default initialization values
        pk_h, pt_k, pl_h, pt_l = 0.0, 0.0, 0.0, 0.0
        
        if engine_type == 0:
            pk_h = round(float(MODELS['pred_kwh_per_hour'].predict(feature_vector)[0]), 2)
            pt_k = round(float(MODELS['pred_total_kwh'].predict(feature_vector)[0]), 2)
        else:
            pl_h = round(float(MODELS['pred_liters_per_hour'].predict(feature_vector)[0]), 2)
            pt_l = round(float(MODELS['pred_total_liters'].predict(feature_vector)[0]), 2)

        # Synchronize Global State Metrics Cache
        for idx, col in enumerate(feature_cols):
            SYSTEM_STATE[col] = input_data[idx]
            
        SYSTEM_STATE['pred_kwh_per_hour'] = pk_h
        SYSTEM_STATE['pred_total_kwh'] = pt_k
        SYSTEM_STATE['pred_liters_per_hour'] = pl_h
        SYSTEM_STATE['pred_total_liters'] = pt_l

        # Append data parameters directly to Ledger History
        log_entry = SYSTEM_STATE.copy()
        log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LEDGER_HISTORY.insert(0, log_entry)

        return jsonify({
            "success": True,
            "prediction_kwh_per_hour": pk_h, "prediction_total_kwh": pt_k,
            "prediction_liters_per_hour": pl_h, "prediction_total_liters": pt_l
        })
    except Exception as ex:
        return jsonify({"success": False, "error": str(ex)}), 400

@app.route('/api/mesh-grid', methods=['GET'])
def get_mesh_grid():
    # Emulate localized cluster rolling stock assets on the active rail grid network
    base_speed = float(SYSTEM_STATE['speed'])
    base_weight = float(SYSTEM_STATE['weight'])
    engine_type = int(SYSTEM_STATE['engine_type'])
    
    mesh_assets = [
        {"id": "NODE-011", "name": "V-Pilot Express 101", "speed": round(base_speed, 1), "weight": round(base_weight, 1)},
        {"id": "NODE-042", "name": "InterCity Commuter Alpha", "speed": round(base_speed * 0.85, 1), "weight": round(base_weight * 1.2, 1)},
        {"id": "NODE-089", "name": "Heavy Freight Vector East", "speed": round(base_speed * 0.55, 1), "weight": round(base_weight * 2.8, 1)}
    ]
    
    for asset in mesh_assets:
        if engine_type == 0:
            val = round((asset['speed'] * 1.1) + (asset['weight'] * 0.08), 2)
            asset['metrics'] = f"{val} kW/h"
            asset['status'] = "PANTOGRAPH SYNCED"
        else:
            val = round((asset['speed'] * 0.25) + (asset['weight'] * 0.02), 2)
            asset['metrics'] = f"{val} L/h"
            asset['status'] = "COMBUSTION SYNCED"
            
    return jsonify(mesh_assets)

if __name__ == '__main__':
    app.run(debug=True, port=5000)