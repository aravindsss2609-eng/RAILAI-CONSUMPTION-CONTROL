# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

FEATURE_COLS = [
    'engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp',
    'aux_load', 'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff',
    'gear_ratio', 'wheel_diam', 'motor_freq', 'brake_pressure', 'regen',
    'control_override', 'simulation_pass'
]

def calculate_physics_power_vectorized(X):
    """
    Computes theoretical tractive and auxiliary power (kW) using high-fidelity physics.
    Supports both pandas DataFrames and numpy matrices seamlessly.
    """
    if isinstance(X, pd.DataFrame):
        speed = X['speed'].values
        weight = X['weight'].values
        gradient = X['gradient'].values
        passengers = X['passengers'].values
        temp = X['temp'].values
        aux_load = X['aux_load'].values
        headwind = X['headwind'].values
        drag_coeff = X['drag_coeff'].values
        rolling_res = X['rolling_res'].values
        inverter_eff = X['inverter_eff'].values
        regen = X['regen'].values
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        # Index mappings strictly aligned with FEATURE_COLS array
        speed = X_arr[:, 1]
        weight = X_arr[:, 2]
        gradient = X_arr[:, 3]
        passengers = X_arr[:, 5]
        temp = X_arr[:, 6]
        aux_load = X_arr[:, 7]
        headwind = X_arr[:, 8]
        drag_coeff = X_arr[:, 9]
        rolling_res = X_arr[:, 10]
        inverter_eff = X_arr[:, 12]
        regen = X_arr[:, 17]

    # Convert speed units: km/h -> m/s
    v = speed / 3.6  
    v_headwind = headwind / 3.6 
    v_rel = v + v_headwind
    
    # Dynamic Mass: Empty train weight (tons -> kg) + avg passenger weight (80kg)
    mass_kg = (weight * 1000) + (passengers * 80)
    
    # Grade Angle calculation
    theta = np.arctan(gradient / 100.0)
    
    # 1. Grade Resistance Force
    F_grade = mass_kg * 9.81 * np.sin(theta)
    
    # 2. Rolling Resistance Force
    F_roll = mass_kg * 9.81 * rolling_res * np.cos(theta)
    
    # 3. Aerodynamic Drag Force (Frontal area of typical high-speed train ~ 11.0 m²)
    frontal_area = 11.0 
    air_density = 1.225 * (273.15 / (273.15 + temp)) 
    F_drag = 0.5 * air_density * drag_coeff * frontal_area * (v_rel ** 2)
    
    # Total Tractive Force at wheel level
    F_traction = F_grade + F_roll + F_drag
    
    # Mechanical Power at wheels (kW)
    P_wheel_kw = (F_traction * v) / 1000.0
    
    # Dynamic Inverter & Drivetrain Efficiency mapping
    eff_factor = np.where(inverter_eff > 0, inverter_eff / 100.0, 0.95)
    
    # Calculate electrical consumption based on traction or regenerative braking
    P_traction_kw = np.where(
        P_wheel_kw >= 0, 
        P_wheel_kw / eff_factor, 
        P_wheel_kw * eff_factor * (regen / 100.0)
    )
    
    # Total Power = Traction + HVAC/Auxiliary systems
    P_total_kw = P_traction_kw + aux_load
    
    # Safety margin: stationary train consumes only auxiliary power
    P_total_kw = np.where(speed < 1, aux_load, P_total_kw)
    
    return P_total_kw


class PhysicsInformedEstimator:
    """
    Standard sklearn-compatible wrapper. Handles exact deterministic totals calculation 
    internally and formats values cleanly back to the caller application context.
    """
    def __init__(self, rate_model, return_total=False):
        self.rate_model = rate_model
        self.return_total = return_total

    def predict(self, X):
        # Calculate high-fidelity physical baseline
        p_phys = calculate_physics_power_vectorized(X)
        
        # Format and append physics feature vector uniformly
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            
        X_augmented = np.column_stack((X_arr, p_phys))
            
        # Predict the corrected rate
        predicted_rate = self.rate_model.predict(X_augmented)
        
        # Return physical total consumption if requested
        if self.return_total:
            speed = X_arr[:, 1]
            distance = X_arr[:, 4]
                
            travel_time_hours = np.where(speed > 0, distance / speed, 0)
            result = predicted_rate * travel_time_hours
            return result[0] if result.ndim > 0 and len(result) == 1 else result
        
        return predicted_rate[0] if predicted_rate.ndim > 0 and len(predicted_rate) == 1 else predicted_rate


def train_system_estimators():
    try:
        df = pd.read_csv('rail_telemetry_data.csv')
    except FileNotFoundError:
        print("Data source missing. Instantiating pipeline generator...")
        import data_generate
        data_generate.generate_telemetry_dataset()
        df = pd.read_csv('rail_telemetry_data.csv')
    
    X = df[FEATURE_COLS]
    
    # Separate modes
    electric_idx = df['engine_type'] == 0
    diesel_idx = df['engine_type'] == 1
    
    final_serialized_models = {}
    
    targets = {
        'electric': {
            'idx': electric_idx,
            'rate_col': 'pred_kwh_per_hour',
            'total_col': 'pred_total_kwh'
        },
        'diesel': {
            'idx': diesel_idx,
            'rate_col': 'pred_liters_per_hour',
            'total_col': 'pred_total_liters'
        }
    }
    
    for mode, meta in targets.items():
        X_sub = X[meta['idx']]
        y_rate = df[meta['idx']][meta['rate_col']].values
        
        # Calculate physics power for target split
        X_sub_phys = calculate_physics_power_vectorized(X_sub)
        X_sub_augmented = np.column_stack((X_sub.values, X_sub_phys))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub_augmented, y_rate, test_size=0.15, random_state=42
        )
        
        regressor = RandomForestRegressor(
            n_estimators=150,
            max_depth=22,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.8,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"Training physics-informed {mode} estimator...")
        regressor.fit(X_train, y_train)
        
        predictions_rate = regressor.predict(X_test)
        r2 = r2_score(y_test, predictions_rate)
        print(f"[{mode.upper()} ENERGY RATE MODEL] R² Target Accuracy: {r2*100:.3f}%")
        
        final_serialized_models[meta['rate_col']] = PhysicsInformedEstimator(regressor, return_total=False)
        final_serialized_models[meta['total_col']] = PhysicsInformedEstimator(regressor, return_total=True)

    # Save to disk
    with open('rail_ai_models.pkl', 'wb') as f:
        pickle.dump(final_serialized_models, f)
    print("\n[SUCCESS] Physics-Informed estimators saved to rail_ai_models.pkl")

if __name__ == "__main__":
    train_system_estimators()