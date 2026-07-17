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

# Core class reconstruction matching train_model logic to protect data flow
try:
    import train_model
    PhysicsInformedEstimator = train_model.PhysicsInformedEstimator
    calculate_physics_power_vectorized = train_model.calculate_physics_power_vectorized
    logger.info("Physics-informed classes imported from train_model successfully.")
except ImportError:
    # Inline fallback declaration to survive standalone execution without module reference breakage
    calculate_physics_power_vectorized = None
    class PhysicsInformedEstimator:
        def __init__(self, rate_model, return_total=False):
            self.rate_model = rate_model
            self.return_total = return_total
        def predict(self, X):
            p_phys = sys.modules['__main__'].calculate_physics_power_vectorized(X)
            X_arr = np.asarray(X)
            if X_arr.ndim == 1: X_arr = X_arr.reshape(1, -1)
            X_augmented = np.column_stack((X_arr, p_phys))
            predicted_rate = self.rate_model.predict(X_augmented)
            if self.return_total:
                speed, distance = X_arr[:, 1], X_arr[:, 4]
                travel_time = np.where(speed > 0, distance / speed, 0)
                return np.array(predicted_rate * travel_time)
            return np.array(predicted_rate)
    logger.warning("train_model.py binding fell back to internal structure definitions.")

# Force bind references directly into local __main__ scope to satisfy Pickle stream maps
sys.modules['__main__'].PhysicsInformedEstimator = PhysicsInformedEstimator
if calculate_physics_power_vectorized:
    sys.modules['__main__'].calculate_physics_power_vectorized = calculate_physics_power_vectorized

# Robust Custom Unpickler to solve namespace discrepancies cross-platform
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PhysicsInformedEstimator':
            return PhysicsInformedEstimator
        return super().find_class(module, name)

app = Flask(__name__)

SYSTEM_STATE = {
    "engine_type": 0, "speed": 120.0, "weight": 450.0, "gradient": 0.5, "distance": 50.0,
    "passengers": 320, "temp": 24.0, "aux_load": 45.0, "headwind": 15.0, "drag_coeff": 0.28,
    "rolling_res": 0.0015, "adhesion": 0.32, "inverter_eff": 94.0, "gear_ratio": 4.12,
    "wheel_diam": 920.0, "motor_freq": 60.0, "brake_pressure": 420.0, "regen": 28.0,
    "control_override": 1, "simulation_pass": 1,
    
    "pred_kwh_per_hour": 1274.26, "pred_total_kwh": 530.94,
    "pred_liters_per_hour": 0.0, "pred_total_liters": 0.0,
    
    "rate": 1274.26, "total": 530.94,
    "kwh_per_hour": 1274.26, "total_kwh": 530.94,
    "liters_per_hour": 0.0, "total_liters": 0.0,
    
    "grid_spot_price": 0.14, "cabin_humidity": 45.0, "hvac_coefficient": 1.2, "thermal_load_status": "STABLE"
}

LEDGER_HISTORY = []
MODELS = None

def init_inference_engine():
    global MODELS
    model_path = 'rail_ai_models.pkl'
    
    if not os.path.exists(model_path):
        logger.info("Target 'rail_ai_models.pkl' missing. Regenerating estimators...")
        try:
            import train_model
            train_model.train_system_estimators()
        except Exception as e:
            logger.error(f"Critical asset generation failure: {e}")
            
    try:
        with open(model_path, 'rb') as f:
            # Use our safe custom unpickler instead of the standard pickle.load()
            MODELS = SafeUnpickler(f).load()
        logger.info("Industrial ML models unpickled and loaded safely into application memory context.")
    except Exception as e:
        logger.error(f"Critical exception unpickling models asset package: {e}")
        MODELS = None

# Initialize the engine
init_inference_engine()

def perform_prediction(input_params):
    if MODELS is None:
        raise RuntimeError("Inference engine models are currently unavailable.")
        
    feature_cols = [
        'engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp', 'aux_load', 
        'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff', 'gear_ratio', 
        'wheel_diam', 'motor_freq', 'brake_pressure', 'regen', 'control_override', 'simulation_pass'
    ]
    
    raw_vector = []
    for col in feature_cols:
        val = input_params.get(col)
        if val is None:
            val = SYSTEM_STATE.get(col, 0.0)
        raw_vector.append(float(val))
        
    feature_vector = np.array([raw_vector], dtype=float)
    engine_type = int(input_params.get('engine_type', SYSTEM_STATE.get('engine_type', 0)))
    
    prediction_results = {
        "pred_kwh_per_hour": 0.0, "pred_total_kwh": 0.0,
        "pred_liters_per_hour": 0.0, "pred_total_liters": 0.0,
    }
    
    processed_vector = feature_vector
    try:
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as sf:
                local_scaler = pickle.load(sf)
                processed_vector = local_scaler.transform(feature_vector)
        elif isinstance(MODELS, dict) and 'scaler' in MODELS and MODELS['scaler'] is not None:
            processed_vector = MODELS['scaler'].transform(feature_vector)
    except Exception as scaler_err:
        logger.warning(f"Scaling pipeline bypassed or unavailable: {scaler_err}")
        processed_vector = feature_vector

    def extract_pred(model_key, data):
        if isinstance(MODELS, dict) and model_key in MODELS:
            estimator = MODELS[model_key]
        elif hasattr(MODELS, 'predict'):
            estimator = MODELS
        else:
            raise KeyError(f"Target estimator key '{model_key}' not resolved in model structure.")
            
        prediction = estimator.predict(data)
        
        if hasattr(prediction, '__len__') or isinstance(prediction, np.ndarray):
            val = prediction[0]
        else:
            val = prediction
        return round(float(val), 2)

    try:
        if engine_type == 0:  # Electric Mode
            prediction_results["pred_kwh_per_hour"] = extract_pred('pred_kwh_per_hour', processed_vector)
            prediction_results["pred_total_kwh"] = extract_pred('pred_total_kwh', processed_vector)
            
            prediction_results["rate"] = prediction_results["pred_kwh_per_hour"]
            prediction_results["total"] = prediction_results["pred_total_kwh"]
            prediction_results["kwh_per_hour"] = prediction_results["pred_kwh_per_hour"]
            prediction_results["total_kwh"] = prediction_results["pred_total_kwh"]
        else:  # Diesel Mode
            prediction_results["pred_liters_per_hour"] = extract_pred('pred_liters_per_hour', processed_vector)
            prediction_results["pred_total_liters"] = extract_pred('pred_total_liters', processed_vector)
            
            prediction_results["rate"] = prediction_results["pred_liters_per_hour"]
            prediction_results["total"] = prediction_results["pred_total_liters"]
            prediction_results["liters_per_hour"] = prediction_results["pred_liters_per_hour"]
            prediction_results["total_liters"] = prediction_results["pred_total_liters"]
    except Exception as pred_err:
        logger.error(f"Error encountered inside internal prediction handlers: {pred_err}")
        raise pred_err

    return prediction_results

# --- Routes ---
@app.route('/')
def route_dashboard(): return render_template('index.html', system_state=SYSTEM_STATE)

@app.route('/asset-health')
def route_asset_health(): return render_template('asset_health.html', system_state=SYSTEM_STATE)

@app.route('/energy')
def route_energy(): return render_template('energy.html', system_state=SYSTEM_STATE)

@app.route('/security')
def route_security(): return render_template('security.html', system_state=SYSTEM_STATE)

@app.route('/traffic')
def route_traffic(): return render_template('traffic.html', system_state=SYSTEM_STATE)

@app.route('/analytics')
def route_analytics(): return render_template('analytics.html', system_state=SYSTEM_STATE)

@app.route('/charts')
def route_charts(): return render_template('charts.html', system_state=SYSTEM_STATE)

@app.route('/history')
def route_history(): return render_template('history.html', ledger=LEDGER_HISTORY)

@app.route('/thermo_passenger')
def route_thermo_passenger(): return render_template('thermo_passenger.html', system_state=SYSTEM_STATE)

@app.route('/predict_page')
def route_predict_page(): return render_template('predict.html', system_state=SYSTEM_STATE)

@app.route('/predict', methods=['GET', 'POST'])
def execute_inference():
    global SYSTEM_STATE
    if request.method == 'GET':
        return render_template('predict.html', system_state=SYSTEM_STATE)

    try:
        req_data = request.get_json(force=True) or {}
        predictions = perform_prediction(req_data)

        feature_cols = [
            'engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp', 'aux_load', 
            'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff', 'gear_ratio', 
            'wheel_diam', 'motor_freq', 'brake_pressure', 'regen', 'control_override', 'simulation_pass'
        ]

        for col in feature_cols:
            if col in req_data:
                SYSTEM_STATE[col] = float(req_data[col]) if col != 'engine_type' else int(req_data[col])

        SYSTEM_STATE.update(predictions)

        log_entry = SYSTEM_STATE.copy()
        log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LEDGER_HISTORY.insert(0, log_entry)

        return jsonify({"success": True, **predictions})

    except Exception as e:
        logger.error(f"Execution handling logic runtime fault: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/get-system-state', methods=['GET'])
def get_system_state(): return jsonify(SYSTEM_STATE)

@app.route('/api/get-history-ledger', methods=['GET'])
def get_history_ledger(): return jsonify(LEDGER_HISTORY)

@app.route('/api/flush-history', methods=['POST'])
def flush_history_ledger():
    global LEDGER_HISTORY
    LEDGER_HISTORY.clear()
    return jsonify({"success": True, "message": "History storage cleared."})

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