# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def train_system_estimators():
    try:
        df = pd.read_csv('rail_telemetry_data.csv')
    except FileNotFoundError:
        print("Data source missing. Instantiating pipeline generator...")
        import data_generate
        data_generate.generate_telemetry_dataset()
        df = pd.read_csv('rail_telemetry_data.csv')

    # Declare feature matrix mappings
    feature_cols = [
        'engine_type', 'speed', 'weight', 'gradient', 'distance', 'passengers', 'temp',
        'aux_load', 'headwind', 'drag_coeff', 'rolling_res', 'adhesion', 'inverter_eff',
        'gear_ratio', 'wheel_diam', 'motor_freq', 'brake_pressure', 'regen',
        'control_override', 'simulation_pass'
    ]
    
    X = df[feature_cols]
    
    # Split datasets into internal execution vectors
    electric_idx = df['engine_type'] == 0
    diesel_idx = df['engine_type'] == 1
    
    # Decoupled Core Storage Models
    models = {}
    metrics_log = {}
    
    # Configurations configuration framework
    targets = {
        'electric': {
            'idx': electric_idx,
            'outputs': ['pred_kwh_per_hour', 'pred_total_kwh']
        },
        'diesel': {
            'idx': diesel_idx,
            'outputs': ['pred_liters_per_hour', 'pred_total_liters']
        }
    }
    
    for mode, target_meta in targets.items():
        X_sub = X[target_meta['idx']]
        
        for out_col in target_meta['outputs']:
            y_sub = df[target_meta['idx']][out_col]
            
            # FIXED: Corrected test_test_split to test_size
            X_train, X_test, y_train, y_test = train_test_split(
                X_sub, y_sub, test_size=0.2, random_state=42
            )
            
            # Use regularization to optimize training and maximize accuracy
            regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            
            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_test)
            r2 = r2_score(y_test, predictions)
            
            models[out_col] = regressor
            metrics_log[out_col] = r2
            print(f"Target Evaluation Metrics [{out_col}] -> R² Accuracy Score: {r2:.4f}")
            
    # Serialize complete tracking artifact
    with open('rail_ai_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    print("Serialized modern high-fidelity estimators successfully saved inside rail_ai_models.pkl")

if __name__ == "__main__":
    train_system_estimators()