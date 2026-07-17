# app.py
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

# Ensure physics-informed custom estimators hook smoothly into main execution context to allow safe unpickling
try:
    import train_model
    sys.modules['__main__'].PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    sys.modules['__main__'].calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    
    PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    logger.info("Physics-informed classes injected into __main__ successfully.")
except ImportError:
    PhysicsInformedEstimator = None
    calculate_physics_power_vectorized = None
    logger.warning("train_model.py dependencies missing. Proceeding with fallback unpickling configurations.")

app = Flask(__name__)

# System state pre-populated with UI fallback permutations to prevent runtime display errors
SYSTEM_STATE = {
    "engine_type": 0, "speed": 120.0, "weight": 450.0, "gradient": 0.5, "distance": 50.0,
    "passengers": 320, "temp": 24.0, "aux_load": 45.0, "headwind": 15.0, "drag_coeff": 0.28,
    "rolling_res": 0.0015, "adhesion": 0.32, "inverter_eff": 94.0, "gear_ratio": 4.12,
    "wheel_diam": 920.0, "motor_freq": 60.0, "brake_pressure": 420.0, "regen": 28.0,
    "control_override": 1, "simulation_pass": 1,
    
    # Primary Targets
    "pred_kwh_per_hour": 1274.26, "pred_total_kwh": 530.94,
    "pred_liters_per_hour": 0.0, "pred_total_liters": 0.0,
    
    # Unified Fallback Naming Aliases for Templates / JS Files
    "rate": 1274.26,
    "total": 530.94,
    "kwh_per_hour": 1274.26,
    "total_kwh": 530.94,
    "liters_per_hour": 0.0,
    "total_liters": 0.0,
    
    # Environmental Auxiliaries
    "grid_spot_price": 0.14, "cabin_humidity": 45.0, "hvac_coefficient": 1.2, "thermal_load_status": "STABLE"
}

# Persistent in-memory ledger tracking user prediction historical commits
LEDGER_HISTORY = []

# Container for ML models & scalers
MODELS = None

def init_inference_engine():
    global MODELS
    model_path = 'rail_ai_models.pkl'
    
    # Auto-train if missing
    if not os.path.exists(model_path):
        logger.info("Target 'rail_ai_models.pkl' missing. Regenerating estimators via train_model.py...")
        try:
            import train_model
            train_model.train_system_estimators()
        except Exception as e:
            logger.error(f"Critical asset generation failure: {e}")
            
    try:
        with open(model_path, 'rb') as f:
            MODELS = pickle.load(f)
        logger.info("Industrial ML models loaded successfully into global app memory context.")
    except Exception as e:
        logger.error(f"Critical exception unpickling models asset package: {e}")
        MODELS = None

# Initialize upon initialization phase
init_inference_engine()

def perform_prediction(input_params):
    if MODELS is None:
        raise RuntimeError("Inference engine models are currently unavailable.")
        
    feature_cols = [
        'engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp', 'aux_load', 
        'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff', 'gear_ratio', 
        'wheel_diam', 'motor_freq', 'brake_pressure', 'regen', 'control_override', 'simulation_pass'
    ]
    
    # Vector generation fallback parsing
    raw_vector = []
    for col in feature_cols:
        val = input_params.get(col, SYSTEM_STATE.get(col))
        raw_vector.append(float(val))
        
    feature_vector = np.array([raw_vector])
    engine_type = int(input_params.get('engine_type', SYSTEM_STATE.get('engine_type', 0)))
    
    prediction_results = {
        "pred_kwh_per_hour": 0.0, "pred_total_kwh": 0.0,
        "pred_liters_per_hour": 0.0, "pred_total_liters": 0.0,
    }
    
    # Helper to clean model outcomes safely out of native NumPy types
    def extract_pred(model_key, data):
        estimator = MODELS[model_key]
        # Check if dictionary contains standalone scaler alongside models
        if isinstance(MODELS, dict) and 'scaler' in MODELS and MODELS['scaler'] is not None:
            processed_data = MODELS['scaler'].transform(data)
        else:
            processed_data = data
            
        prediction = estimator.predict(processed_data)
        return round(float(prediction[0]), 2)

    if engine_type == 0:  # Electric Archetype Mode
        prediction_results["pred_kwh_per_hour"] = extract_pred('pred_kwh_per_hour', feature_vector)
        prediction_results["pred_total_kwh"] = extract_pred('pred_total_kwh', feature_vector)
        
        # Sync structural fallbacks
        prediction_results["rate"] = prediction_results["pred_kwh_per_hour"]
        prediction_results["total"] = prediction_results["pred_total_kwh"]
        prediction_results["kwh_per_hour"] = prediction_results["pred_kwh_per_hour"]
        prediction_results["total_kwh"] = prediction_results["pred_total_kwh"]
        prediction_results["liters_per_hour"] = 0.0
        prediction_results["total_liters"] = 0.0
    else:  # Combustion Diesel Mode
        prediction_results["pred_liters_per_hour"] = extract_pred('pred_liters_per_hour', feature_vector)
        prediction_results["pred_total_liters"] = extract_pred('pred_total_liters', feature_vector)
        
        # Sync structural fallbacks
        prediction_results["rate"] = prediction_results["pred_liters_per_hour"]
        prediction_results["total"] = prediction_results["pred_total_liters"]
        prediction_results["liters_per_hour"] = prediction_results["pred_liters_per_hour"]
        prediction_results["total_liters"] = prediction_results["pred_total_liters"]
        prediction_results["kwh_per_hour"] = 0.0
        prediction_results["total_kwh"] = 0.0

    return prediction_results

# --- UI Server-Side Template Rendering Routes ---
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

@app.route('/predict', methods=['GET', 'POST'])
def execute_inference():
    global SYSTEM_STATE
    if request.method == 'GET':
        return render_template('predict.html', system_state=SYSTEM_STATE)

    try:
        req_data = request.get_json(force=True) or {}
        predictions = perform_prediction(req_data)

        # Sync update input parameters cleanly into state
        feature_cols = [
            'engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp', 'aux_load', 
            'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff', 'gear_ratio', 
            'wheel_diam', 'motor_freq', 'brake_pressure', 'regen', 'control_override', 'simulation_pass'
        ]

        for col in feature_cols:
            if col in req_data:
                SYSTEM_STATE[col] = float(req_data[col]) if col != 'engine_type' else int(req_data[col])

        # Push calculated outputs into global system state parameters
        SYSTEM_STATE.update(predictions)

        # Inject unified timestamp logging map object back into the historical stack array
        log_entry = SYSTEM_STATE.copy()
        log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LEDGER_HISTORY.insert(0, log_entry)

        return jsonify({"success": True, **predictions})

    except Exception as e:
        logger.error(f"Execution handling logic runtime fault: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

# --- Core API Telemetry Sync Pipes ---
@app.route('/api/get-system-state', methods=['GET'])
def get_system_state():
    return jsonify(SYSTEM_STATE)

@app.route('/api/get-history-ledger', methods=['GET'])
def get_history_ledger():
    return jsonify(LEDGER_HISTORY)

@app.route('/api/flush-history', methods=['POST'])
def flush_history_ledger():
    global LEDGER_HISTORY
    LEDGER_HISTORY.clear()
    return jsonify({"success": True, "message": "History tracking storage cleared successfully."})

# --- Global Navigation Patch Injection Processing Filter ---
@app.after_request
def inject_global_navigation(response):
    if response.mimetype != 'text/html':
        return response
        
    html_content = response.get_data(as_text=True)
    current_path = request.path

    nav_links = [
        ("/", "Dashboard", "fa-gauge"),
        ("/energy", "Energy", "fa-bolt"),
        ("/thermo_passenger", "Thermo", "fa-temperature-half"),
        ("/predict", "Predict", "fa-robot"),
        ("/predict_page", "AI Lab", "fa-microscope"),
        ("/traffic", "Traffic", "fa-traffic-light"),
        ("/analytics", "Analytics", "fa-chart-line"),
        ("/history", "History", "fa-history")
    ]

    links_html = []
    for path, label, icon in nav_links:
        is_active = (path == current_path or (path == "/predict" and current_path == "/predict_page"))
        class_str = "text-emerald-400 font-bold border-b border-emerald-500 pb-1" if is_active else "text-slate-400 hover:text-emerald-400 transition"
        icon_html = f'<i class="fa-solid {icon} text-[10px] mr-1.5"></i>' if icon else ''
        links_html.append(f'<a href="{path}" class="{class_str} flex items-center">{icon_html}{label}</a>')

    unified_navbar = f"""
    <nav class="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex justify-between items-center">
        <div class="flex items-center space-x-3">
            <div class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
            <span class="text-lg font-black text-slate-100 tracking-wider">RAIL<span class="text-emerald-500">AI</span> CONTROL</span>
        </div>
        <div class="flex gap-x-6 text-[11px] font-mono uppercase tracking-widest">
            {" ".join(links_html)}
        </div>
    </nav>
    """
    
    patched_html = re.sub(r'<nav.*?</nav>', unified_navbar, html_content, flags=re.DOTALL)
    response.set_data(patched_html)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)