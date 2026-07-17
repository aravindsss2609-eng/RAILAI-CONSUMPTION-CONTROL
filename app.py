from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import sys
import random
import re
from datetime import datetime

# Import custom physics-informed classes to allow pickle to safely deserialize them
try:
    import train_model
    # --- Bulletproof Pickle Namespace Alignment ---
    sys.modules['__main__'].PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    sys.modules['__main__'].calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
except ImportError:
    PhysicsInformedEstimator = None
    calculate_physics_power_vectorized = None

app = Flask(__name__)

# Expanded System Memory Cache Arrays (State Ledger)
SYSTEM_STATE = {
    "engine_type": 0, "speed": 120.0, "weight": 450.0, "gradient": 0.5, "distance": 50.0,
    "passengers": 320, "temp": 24.0, "aux_load": 45.0, "headwind": 15.0, "drag_coeff": 0.28,
    "rolling_res": 0.0015, "adhesion": 0.32, "inverter_eff": 94.0, "gear_ratio": 4.12,
    "wheel_diam": 920.0, "motor_freq": 60.0, "brake_pressure": 420.0, "regen": 0.28,
    "control_override": 1, "simulation_pass": 1,
    "pred_kwh_per_hour": 182.4, "pred_total_kwh": 76.0,
    "pred_liters_per_hour": 0.0, "pred_total_liters": 0.0,
    "grid_spot_price": 0.14, "cabin_humidity": 45.0, "hvac_coefficient": 1.2, "thermal_load_status": "STABLE"
}

LEDGER_HISTORY = []
MODELS = None

def init_inference_engine():
    global MODELS
    model_path = 'rail_ai_models.pkl'
    if not os.path.exists(model_path):
        import train_model
        train_model.train_system_estimators()
    
    global PhysicsInformedEstimator, calculate_physics_power_vectorized
    import train_model
    sys.modules['__main__'].PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    sys.modules['__main__'].calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    with open(model_path, 'rb') as f:
        MODELS = pickle.load(f)

init_inference_engine()

# --- Standard Core Page Mappings ---
@app.route('/')
def route_dashboard(): return render_template('index.html')

# --- Add these 4 missing route mappings to app.py ---

@app.route('/asset-health')
def route_asset_health(): 
    return render_template('asset_health.html')

@app.route('/energy')
def route_energy(): 
    return render_template('energy.html')

@app.route('/security')
def route_security(): 
    return render_template('security.html')

@app.route('/traffic')
def route_traffic(): 
    return render_template('traffic.html')

@app.route('/predict_page')
def route_predict_page(): return render_template('predict.html')

@app.route('/analytics')
def route_analytics(): return render_template('analytics.html')

@app.route('/charts')
def route_charts(): return render_template('charts.html')

@app.route('/history')
def route_history(): return render_template('history.html')

# --- Four New Industrial Module Mappings ---
# --- Bridge API Endpoints for New Industrial Modules ---

@app.route('/api/get-asset-health', methods=['GET'])
def get_asset_health():
    # Maps system state to the Asset Health Dashboard metrics
    return jsonify({
        "mechanical_integrity": 98.4,
        "vibration_g": 0.05,
        "thermal_stability": SYSTEM_STATE.get("temp", 30),
        "predictive_maintenance_km": 12000,
        "wear_metrics": {"inverter": 99, "bearing": 82, "cooling": 96}
    })

@app.route('/api/get-energy-metrics', methods=['GET'])
def get_energy_metrics():
    # Returns data for the Energy Consumption Matrix
    return jsonify({
        "current_draw": SYSTEM_STATE.get("pred_kwh_per_hour", 0),
        "recovery": 12.2,
        "distribution": {"traction": 74, "climate": 12, "aux": 14},
        "history": [170, 175, 180, 178, 182, SYSTEM_STATE.get("pred_kwh_per_hour", 182.4)]
    })

@app.route('/api/get-traffic-status', methods=['GET'])
def get_traffic_status():
    # Returns predictive load data for the Traffic & Signaling page
    return jsonify({
        "system_throughput": 98.2,
        "segment_load": [45, 78, 30, 92, 55, 40],
        "node_status": {"A-1": "CLEAR", "B-4": "CAUTION", "C-9": "CLEAR"}
    })
@app.route('/api/get-security-status', methods=['GET'])
def get_security_status():
    # Returns current security telemetry
    return jsonify({
        "encryption_status": "AES-256 ENCRYPTION ACTIVE",
        "firewall_nodes": "48/48",
        "data_integrity": "99.999%",
        "threat_log": [
            {"time": "19:45:02", "msg": "Handshake Validated"},
            {"time": "19:46:15", "msg": "Node Sync Locked"}
        ]
    })
# --- Operational API Framework (Retained Core) ---
@app.route('/api/get-system-state', methods=['GET'])
def get_system_state():
    SYSTEM_STATE["grid_spot_price"] = round(max(0.06, min(0.35, SYSTEM_STATE["grid_spot_price"] + random.uniform(-0.01, 0.01))), 2)
    return jsonify(SYSTEM_STATE)

@app.route('/api/get-history-ledger', methods=['GET'])
def get_history_ledger(): return jsonify(LEDGER_HISTORY)

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
        feature_cols = ['engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp', 'aux_load', 'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff', 'gear_ratio', 'wheel_diam', 'motor_freq', 'brake_pressure', 'regen', 'control_override', 'simulation_pass']
        input_data = [float(req.get(col, SYSTEM_STATE.get(col))) for col in feature_cols]
        feature_vector = np.array([input_data])
        
        engine_type = int(req.get('engine_type', 0))
        pk_h, pt_k, pl_h, pt_l = 0.0, 0.0, 0.0, 0.0
        
        if engine_type == 0:
            pk_h = round(float(MODELS['pred_kwh_per_hour'].predict(feature_vector)[0]), 2)
            pt_k = round(float(MODELS['pred_total_kwh'].predict(feature_vector)[0]), 2)
        else:
            pl_h = round(float(MODELS['pred_liters_per_hour'].predict(feature_vector)[0]), 2)
            pt_l = round(float(MODELS['pred_total_liters'].predict(feature_vector)[0]), 2)

        for idx, col in enumerate(feature_cols): SYSTEM_STATE[col] = input_data[idx]
        SYSTEM_STATE.update({'pred_kwh_per_hour': pk_h, 'pred_total_kwh': pt_k, 'pred_liters_per_hour': pl_h, 'pred_total_liters': pt_l})

        log_entry = SYSTEM_STATE.copy()
        log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LEDGER_HISTORY.insert(0, log_entry)
        return jsonify({"success": True, "prediction_kwh_per_hour": pk_h, "prediction_total_kwh": pt_k, "prediction_liters_per_hour": pl_h, "prediction_total_liters": pt_l})
    except Exception as ex: return jsonify({"success": False, "error": str(ex)}), 400

# --- Global Navigation Patch Engine (Dynamic Injection) ---
# --- Global Navigation Patch Engine (Dynamic Injection) ---
@app.after_request
def inject_global_navigation(response):
    if response.mimetype != 'text/html': return response
    html_content = response.get_data(as_text=True)
    current_path = request.path

    nav_links = [
        ("/", "Dashboard", "fa-gauge"),
        ("/asset-health", "Asset Health", "fa-heart-pulse"),
        ("/energy", "Energy", "fa-bolt"),
        ("/security", "Security", "fa-shield"),
        ("/thermo_passenger", "Thermo_Passenger", "fa-temperature-half"),
        ("/predict", "Predict", "fa-robot"),
        ("/traffic", "Traffic", "fa-traffic-light"),
        ("/analytics", "Analytics", "fa-chart-line"),
        ("/history", "History", "fa-history"),
        ("/predict_page", "AI Lab", "fa-microscope")
    ]
    # ... rest of your function

    links_html = []
    for path, label, icon in nav_links:
        is_active = (path == current_path)
        class_str = "text-emerald-400 font-bold border-b border-emerald-500 pb-1" if is_active else "text-slate-400 hover:text-emerald-400 transition"
        icon_html = f'<i class="fa-solid {icon} text-[10px] mr-1"></i>' if icon else ''
        links_html.append(f'<a href="{path}" class="{class_str} flex items-center">{icon_html}{label}</a>')

    unified_navbar = f"""
    <nav class="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center space-x-3">
            <div class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
            <span class="text-lg font-black text-slate-100 tracking-wider">RAIL<span class="text-emerald-500">AI</span> CONTROL</span>
        </div>
        <div class="flex gap-x-6 text-xs font-mono uppercase tracking-widest">
            {" ".join(links_html)}
        </div>
    </nav>
    """
    patched_html = re.sub(r'<nav.*?</nav>', unified_navbar, html_content, flags=re.DOTALL)
    response.set_data(patched_html)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)