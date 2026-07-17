# data_generate.py
import numpy as np
import pandas as pd

def generate_telemetry_dataset(num_samples=15000):
    np.random.seed(42)
    
    # 1. Sample uniformly across standard operational windows
    engine_type = np.random.choice([0, 1], size=num_samples) # 0: Electric, 1: Diesel
    speed = np.random.uniform(40, 250, size=num_samples)     # km/h
    weight = np.random.uniform(50, 2500, size=num_samples)    # Tons
    gradient = np.random.uniform(-6, 6, size=num_samples)    # %
    distance = np.random.uniform(1, 1000, size=num_samples)  # km
    passengers = np.random.randint(0, 800, size=num_samples)
    temp = np.random.uniform(-20, 50, size=num_samples)      # °C
    aux_load = np.random.uniform(10, 100, size=num_samples)   # kW (HVAC, lighting)
    headwind = np.random.uniform(0, 60, size=num_samples)    # km/h
    drag_coeff = np.random.uniform(0.15, 0.40, size=num_samples)
    rolling_res = np.random.uniform(0.0010, 0.0025, size=num_samples)
    adhesion = np.random.uniform(0.20, 0.45, size=num_samples)
    inverter_eff = np.random.uniform(85, 98, size=num_samples) # %
    gear_ratio = np.random.uniform(3.0, 6.0, size=num_samples)
    wheel_diam = np.random.uniform(800, 1100, size=num_samples) # mm
    motor_freq = np.random.uniform(30, 90, size=num_samples)   # Hz
    brake_pressure = np.random.uniform(300, 600, size=num_samples) # kPa
    regen = np.random.uniform(0.10, 0.45, size=num_samples)
    control_override = np.ones(num_samples)
    simulation_pass = np.ones(num_samples)

    # 2. Physics Core Math Interpolation Engine
    # Effective Mass (Train weight + average human weight approximation)
    effective_mass_kg = (weight + (passengers * 0.075)) * 1000 
    v_ms = speed / 3.6
    v_wind_ms = headwind / 3.6
    
    # Force Vectors: Gravity + Rolling Resistance + Aerodynamic Drag
    f_gravity = effective_mass_kg * 9.81 * (gradient / 100)
    f_rolling = effective_mass_kg * 9.81 * rolling_res
    f_aero = 0.5 * 1.225 * drag_coeff * 12.0 * (v_ms + v_wind_ms)**2
    
    f_total_traction = f_gravity + f_rolling + f_aero
    # Floor traction to maintain low baseline mechanical upkeep energy when descending
    f_total_traction = np.where(f_total_traction < 0, 5000.0, f_total_traction)
    
    # Mechanical Power Output (Watts)
    p_mechanical_kw = (f_total_traction * v_ms) / 1000.0
    p_total_kw = p_mechanical_kw + aux_load
    
    # Time window spent traversing block segment
    hours_spent = distance / speed
    
    # Output Target Structuring
    pred_kwh_per_hour = []
    pred_total_kwh = []
    pred_liters_per_hour = []
    pred_total_liters = []
    
    for i in range(num_samples):
        # Apply Thermal Efficiency Delays
        temp_penalty = 1.0 + (max(0, temp[i] - 25) * 0.005) + (max(0, -5 - temp[i]) * 0.008)
        
        if engine_type[i] == 0:
            # Electric Engine Dynamics
            inv_factor = 100.0 / inverter_eff[i]
            regen_savings = regen[i] if gradient[i] < 0 else 0.0
            
            hourly_kw = max(20.0, p_total_kw[i] * inv_factor * temp_penalty * (1.0 - regen_savings))
            total_kwh = hourly_kw * hours_spent[i]
            
            # Stochastic real-world noise variance (~0.5% standard deviation)
            hourly_kw += np.random.normal(0, hourly_kw * 0.005)
            total_kwh += np.random.normal(0, total_kwh * 0.005)
            
            pred_kwh_per_hour.append(hourly_kw)
            pred_total_kwh.append(total_kwh)
            pred_liters_per_hour.append(0.0)
            pred_total_liters.append(0.0)
        else:
            # Diesel Combustion Dynamics (Approx. 38% baseline thermodynamic brake efficiency)
            # 1 Liter Diesel ~ 10 kWh energy capacity
            hourly_liters = max(5.0, (p_total_kw[i] * temp_penalty) / (10.0 * 0.38))
            total_liters = hourly_liters * hours_spent[i]
            
            hourly_liters += np.random.normal(0, hourly_liters * 0.005)
            total_liters += np.random.normal(0, total_liters * 0.005)
            
            pred_kwh_per_hour.append(0.0)
            pred_total_kwh.append(0.0)
            pred_liters_per_hour.append(hourly_liters)
            pred_total_liters.append(total_liters)

    df = pd.DataFrame({
        'engine_type': engine_type, 'speed': speed, 'weight': weight, 'gradient': gradient,
        'distance': distance, 'passengers': passengers, 'temp': temp, 'aux_load': aux_load,
        'headwind': headwind, 'drag_coeff': drag_coeff, 'rolling_res': rolling_res,
        'adhesion': adhesion, 'inverter_eff': inverter_eff, 'gear_ratio': gear_ratio,
        'wheel_diam': wheel_diam, 'motor_freq': motor_freq, 'brake_pressure': brake_pressure,
        'regen': regen, 'control_override': control_override, 'simulation_pass': simulation_pass,
        'pred_kwh_per_hour': pred_kwh_per_hour, 'pred_total_kwh': pred_total_kwh,
        'pred_liters_per_hour': pred_liters_per_hour, 'pred_total_liters': pred_total_liters
    })
    
    df.to_csv('rail_telemetry_data.csv', index=False)
    print("Dataset initialized inside rail_telemetry_data.csv")

if __name__ == "__main__":
    generate_telemetry_dataset()