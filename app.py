from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import sys
import random
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
    
    # --- Extended Innovative Structural States ---
    "grid_spot_price": 0.14,      # USD per kWh
    "cabin_humidity": 45.0,       # Relative Humidity %
    "hvac_coefficient": 1.2,      # Performance multiplier
    "thermal_load_status": "STABLE"
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
def route_dashboard(): 
    return render_template('index.html')

@app.route('/predict_page')
def route_predict_page(): 
    return render_template('predict.html')

@app.route('/analytics')
def route_analytics(): 
    return render_template('analytics.html')

@app.route('/charts')
def route_charts(): 
    return render_template('charts.html')

@app.route('/history')
def route_history(): 
    return render_template('history.html')

# --- New Innovative Page Mappings (2026 Fleet Standard) ---
@app.route('/mesh-matrix')
def route_mesh_matrix(): 
    return render_template('mesh_matrix.html')

@app.route('/grid-balancing')
def route_grid_balancing(): 
    return render_template('grid_balancing.html')

@app.route('/thermo-passenger')
def route_thermo_passenger(): 
    return render_template('thermo_passenger.html')

# --- Real-Time Operational API Framework ---
@app.route('/api/get-system-state', methods=['GET'])
def get_system_state():
    SYSTEM_STATE["grid_spot_price"] = round(max(0.06, min(0.35, SYSTEM_STATE["grid_spot_price"] + random.uniform(-0.01, 0.01))), 2)
    passenger_density = float(SYSTEM_STATE["passengers"])
    ambient_temp = float(SYSTEM_STATE["temp"])
    if passenger_density > 400 or ambient_temp > 32:
        SYSTEM_STATE["thermal_load_status"] = "HIGH OVERHEAD"
    else:
        SYSTEM_STATE["thermal_load_status"] = "STABLE"
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
        feature_cols = [
            'engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp',
            'aux_load', 'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff',
            'gear_ratio', 'wheel_diam', 'motor_freq', 'brake_pressure', 'regen',
            'control_override', 'simulation_pass'
        ]
        
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

        for idx, col in enumerate(feature_cols):
            SYSTEM_STATE[col] = input_data[idx]
            
        SYSTEM_STATE['pred_kwh_per_hour'] = pk_h
        SYSTEM_STATE['pred_total_kwh'] = pt_k
        SYSTEM_STATE['pred_liters_per_hour'] = pl_h
        SYSTEM_STATE['pred_total_liters'] = pt_l

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
    base_speed = float(SYSTEM_STATE['speed'])
    base_weight = float(SYSTEM_STATE['weight'])
    engine_type = int(SYSTEM_STATE['engine_type'])
    
    mesh_assets = [
        {"id": "NODE-011", "name": "V-Pilot Express 101", "speed": round(base_speed, 1), "weight": round(base_weight, 1), "lat": 40.7128, "lng": -74.0060},
        {"id": "NODE-042", "name": "InterCity Commuter Alpha", "speed": round(base_speed * 0.85, 1), "weight": round(base_weight * 1.2, 1), "lat": 40.7589, "lng": -73.9851},
        {"id": "NODE-089", "name": "Heavy Freight Vector East", "speed": round(base_speed * 0.55, 1), "weight": round(base_weight * 2.8, 1), "lat": 40.7306, "lng": -73.9352}
    ]
    
    for asset in mesh_assets:
        if engine_type == 0:
            val = round((asset['speed'] * 1.1) + (asset['weight'] * 0.08), 2)
            asset['metrics'] = f"{val} kW/h"
            asset['status'] = "PANTOGRAPH SYNCED"
            asset['regen_yield'] = round(val * float(SYSTEM_STATE['regen']), 2)
        else:
            val = round((asset['speed'] * 0.25) + (asset['weight'] * 0.02), 2)
            asset['metrics'] = f"{val} L/h"
            asset['status'] = "COMBUSTION SYNCED"
            asset['regen_yield'] = 0.0
            
    return jsonify(mesh_assets)

# --- Global Navigation Patch Engine Middleware ---
@app.after_request
def inject_global_navigation(response):
    """Intercepts and updates old navigation structures dynamically across all views."""
    if response.mimetype != 'text/html':
        return response

    html_content = response.get_data(as_text=True)
    current_path = request.path

    # Define all available operational routes 
    nav_links = [
        ("/", "Dashboard System", None),
        ("/analytics", "Live Analytics", None),
        ("/charts", "Comparative Charts", None),
        ("/mesh-matrix", "Fleet Mesh", "fa-map-location-dot"),
        ("/grid-balancing", "Grid Balance", "fa-scale-balanced"),
        ("/thermo-passenger", "Thermo System", "fa-temperature-half"),
        ("/history", "Ledger History", None),
        ("/predict_page", "AI Predictor Laboratory", None)
    ]

    # Dynamically build unified navigation elements with proper styling states
    links_html = []
    for path, label, icon in nav_links:
        is_active = (path == current_path) or (path == '/' and current_path == '')
        class_str = "text-emerald-400 font-bold border-b border-emerald-500 pb-1" if is_active else "text-slate-400 hover:text-emerald-400 transition"
        icon_html = f'<i class="fa-solid {icon} text-[10px]"></i> ' if icon else ''
        links_html.append(f'<a href="{path}" class="{class_str} flex items-center gap-1">{icon_html}{label}</a>')

    unified_navbar = f"""
    <!-- Navigation Header Generated Dynamically -->
    <nav class="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex flex-col md:flex-row justify-between items-center gap-4">
        <div class="flex items-center space-x-3">
            <div class="w-3 h-3 rounded-full bg-emerald-500 animate-ping"></div>
            <span class="text-lg font-black tracking-wider text-slate-100">RAIL<span class="text-emerald-500">AI</span> CONTROL</span>
        </div>
        <div class="flex flex-wrap justify-center gap-x-6 gap-y-2 text-xs font-mono uppercase tracking-widest">
            {" ".join(links_html)}
        </div>
    </nav>
    """

    # Seamlessly overwrite legacy or alternative navbar tags
    import re
    if "<nav" in html_content:
        patched_html = re.sub(r'<nav.*?</nav>', unified_navbar, html_content, flags=re.DOTALL)
        response.set_data(patched_html)
    elif "<body" in html_content:
        # Prepend to top of body if layout does not contain an existing <nav> block
        patched_html = re.sub(r'(<body[^>]*>)', r'\1' + unified_navbar, html_content, count=1)
        response.set_data(patched_html)

    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)