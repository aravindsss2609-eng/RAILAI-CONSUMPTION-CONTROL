from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import sys
import re
import logging
from datetime import datetime

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import custom physics-informed classes for enhanced modeling
try:
    import train_model
    sys.modules['__main__'].PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    sys.modules['__main__'].calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    logger.info("Physics-informed classes imported successfully.")
except ImportError:
    PhysicsInformedEstimator = None
    calculate_physics_power_vectorized = None
    logger.warning("Physics-informed classes not found, proceeding without them.")

app = Flask(__name__)

# Default system state, shared globally
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

# History ledger to keep track of predictions and states
LEDGER_HISTORY = []

# Container for ML models, initialized once
MODELS = None

def init_inference_engine():
    global MODELS
    model_path = 'rail_ai_models.pkl'
    if not os.path.exists(model_path):
        logger.info("Model file not found, training new model.")
        import train_model
        train_model.train_system_estimators()
    try:
        with open(model_path, 'rb') as f:
            MODELS = pickle.load(f)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        MODELS = None

init_inference_engine()

# Helper function to perform prediction with given input parameters
def perform_prediction(input_params):
    if MODELS is None:
        raise RuntimeError("Models are not loaded, prediction cannot be performed.")
    
    feature_cols = ['engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp', 'aux_load', 'headwind', 'drag_coeff',
                    'rolling_res', 'adhesion', 'inverter_eff', 'gear_ratio', 'wheel_diam', 'motor_freq', 'brake_pressure', 'regen', 
                    'control_override', 'simulation_pass']
    feature_vector = np.array([[float(input_params.get(col, SYSTEM_STATE.get(col))) for col in feature_cols]])
    engine_type = int(input_params.get('engine_type', 0))
    
    prediction_results = {
        "pred_kwh_per_hour": 0.0,
        "pred_total_kwh": 0.0,
        "pred_liters_per_hour": 0.0,
        "pred_total_liters": 0.0,
    }
    
    if engine_type == 0:  # Electric engine
        prediction_results["pred_kwh_per_hour"] = round(float(MODELS['pred_kwh_per_hour'].predict(feature_vector)[0]), 2)
        prediction_results["pred_total_kwh"] = round(float(MODELS['pred_total_kwh'].predict(feature_vector)[0]), 2)
    else:  # Combustion engine
        prediction_results["pred_liters_per_hour"] = round(float(MODELS['pred_liters_per_hour'].predict(feature_vector)[0]), 2)
        prediction_results["pred_total_liters"] = round(float(MODELS['pred_total_liters'].predict(feature_vector)[0]), 2)

    return prediction_results

# --- Page Routes ---
@app.route('/')
def route_dashboard():
    return render_template('index.html', system_state=SYSTEM_STATE)

@app.route('/asset-health')
def route_asset_health():
    return render_template('asset_health.html', system_state=SYSTEM_STATE)

@app.route('/energy')
def route_energy():
    return render_template('energy.html', system_state=SYSTEM_STATE)

@app.route('/security')
def route_security():
    return render_template('security.html', system_state=SYSTEM_STATE)

@app.route('/traffic')
def route_traffic():
    return render_template('traffic.html', system_state=SYSTEM_STATE)

@app.route('/analytics')
def route_analytics():
    return render_template('analytics.html', system_state=SYSTEM_STATE)

@app.route('/charts')
def route_charts():
    return render_template('charts.html', system_state=SYSTEM_STATE)

@app.route('/history')
def route_history():
    return render_template('history.html', ledger=LEDGER_HISTORY)

@app.route('/thermo_passenger')
def route_thermo_passenger():
    return render_template('thermo_passenger.html', system_state=SYSTEM_STATE)

@app.route('/predict_page')
def route_predict_page():
    return render_template('predict.html', system_state=SYSTEM_STATE)

# Prediction endpoint: supports GET (view page) and POST (run model)
@app.route('/predict', methods=['GET', 'POST'])
def execute_inference():
    global SYSTEM_STATE
    if request.method == 'GET':
        return render_template('predict.html', system_state=SYSTEM_STATE)

    # POST request handling
    try:
        req_data = request.get_json(force=True)
        predictions = perform_prediction(req_data)

        # Update SYSTEM_STATE with new input and prediction results
        feature_cols = ['engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp', 'aux_load', 'headwind', 'drag_coeff',
                        'rolling_res', 'adhesion', 'inverter_eff', 'gear_ratio', 'wheel_diam', 'motor_freq', 'brake_pressure', 'regen', 'control_override', 'simulation_pass']

        for col in feature_cols:
            if col in req_data:
                SYSTEM_STATE[col] = float(req_data[col]) if col != 'engine_type' else int(req_data[col])

        SYSTEM_STATE.update(predictions)

        # Log the updated state with timestamp at the beginning of ledger
        log_entry = SYSTEM_STATE.copy()
        log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LEDGER_HISTORY.insert(0, log_entry)

        return jsonify({"success": True, **predictions})

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

# --- API Endpoints ---
@app.route('/api/get-system-state', methods=['GET'])
def get_system_state():
    return jsonify(SYSTEM_STATE)

@app.route('/api/get-history-ledger', methods=['GET'])
def get_history_ledger():
    return jsonify(LEDGER_HISTORY)

# --- Global Navigation Injection ---
@app.after_request
def inject_global_navigation(response):
    if response.mimetype != 'text/html':
        return response
    html_content = response.get_data(as_text=True)
    current_path = request.path

    nav_links = [
        ("/", "Dashboard", "fa-gauge"),
        ("/asset-health", "Asset Health", "fa-heart-pulse"),
        ("/energy", "Energy", "fa-bolt"),
        ("/security", "Security", "fa-shield"),
        ("/thermo_passenger", "Thermo", "fa-temperature-half"),
        ("/predict", "Predict", "fa-robot"),
        ("/predict_page", "AI Lab", "fa-microscope"),
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
    # Replace existing <nav>...</nav> with unified navbar in the HTML content
    patched_html = re.sub(r'<nav.*?</nav>', unified_navbar, html_content, flags=re.DOTALL)
    response.set_data(patched_html)
    return response

if __name__ == '__main__':
    # Run with debug=False in production for security; debug=True is useful for development
    app.run(host='0.0.0.0', port=5000, debug=True)
