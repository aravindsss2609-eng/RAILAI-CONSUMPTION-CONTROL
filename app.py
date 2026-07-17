from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import sys
import random
import re
from datetime import datetime

# Import custom physics-informed classes
try:
    import train_model
    sys.modules['__main__'].PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    sys.modules['__main__'].calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
except ImportError:
    PhysicsInformedEstimator = None
    calculate_physics_power_vectorized = None

app = Flask(__name__)

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
    
    with open(model_path, 'rb') as f:
        MODELS = pickle.load(f)

init_inference_engine()

# --- Page Routes ---
@app.route('/')
def route_dashboard(): return render_template('index.html')

@app.route('/asset-health')
def route_asset_health(): return render_template('asset_health.html')

@app.route('/energy')
def route_energy(): return render_template('energy.html')

@app.route('/security')
def route_security(): return render_template('security.html')

@app.route('/traffic')
def route_traffic(): return render_template('traffic.html')

@app.route('/analytics')
def route_analytics(): return render_template('analytics.html')

@app.route('/charts')
def route_charts(): return render_template('charts.html')

@app.route('/history')
def route_history(): return render_template('history.html')

@app.route('/thermo_passenger')
def route_thermo_passenger(): return render_template('thermo_passenger.html')

@app.route('/predict_page')
def route_predict_page(): return render_template('predict.html')

# Fixed route: accepts both GET (to view page) and POST (to run model)
@app.route('/predict', methods=['GET', 'POST'])
def execute_inference():
    global SYSTEM_STATE
    if request.method == 'GET':
        return render_template('predict.html') # Ensure you have a predict.html
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

# --- API Endpoints ---
@app.route('/api/get-system-state', methods=['GET'])
def get_system_state(): return jsonify(SYSTEM_STATE)

@app.route('/api/get-history-ledger', methods=['GET'])
def get_history_ledger(): return jsonify(LEDGER_HISTORY)

# --- Global Navigation ---
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
        ("/thermo_passenger", "Thermo", "fa-temperature-half"),
        ("/predict", "Predict", "fa-robot"),
        ("/traffic", "Traffic", "fa-traffic-light"),
        ("/analytics", "Analytics", "fa-chart-line"),
        ("/history", "History", "fa-history")
    ]

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