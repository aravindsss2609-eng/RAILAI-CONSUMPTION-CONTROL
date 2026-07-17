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
    
    # Scale to percentage (10% to 45%) to perfectly match app.py defaults and train_model.py scaling
    regen = np.random.uniform(10.0, 45.0, size=num_samples) 
    
    control_override = np.ones(num_samples)
    simulation_pass = np.ones(num_samples)

    # 2. Physics Core Math Interpolation Engine (Synchronized with train_model.py)
    effective_mass_kg = (weight * 1000) + (passengers * 80)
    v_ms = speed / 3.6
    v_wind_ms = headwind / 3.6
    v_rel = v_ms + v_wind_ms
    
    theta = np.arctan(gradient / 100.0)
    
    # Force Vectors: Gravity + Rolling Resistance + Aerodynamic Drag
    f_gravity = effective_mass_kg * 9.81 * np.sin(theta)
    f_rolling = effective_mass_kg * 9.81 * rolling_res * np.cos(theta)
    
    # Match frontal area constant (11.0) and include temperature-dependent air density formula
    frontal_area = 11.0
    air_density = 1.225 * (273.15 / (273.15 + temp))
    f_aero = 0.5 * air_density * drag_coeff * frontal_area * (v_rel ** 2)
    
    f_total_traction = f_gravity + f_rolling + f_aero
    
    # Mechanical Power Output (kW)
    p_wheel_kw = (f_total_traction * v_ms) / 1000.0
    
    # Time window spent traversing block segment
    hours_spent = distance / speed
    
    # Output Target Structuring
    pred_kwh_per_hour = []
    pred_total_kwh = []
    pred_liters_per_hour = []
    pred_total_liters = []
    
    for i in range(num_samples):
        # Extract scalar values for the loop element to keep native max() functions happy
        t_val = temp[i]
        p_wheel_val = p_wheel_kw[i]
        aux_val = aux_load[i]
        speed_val = speed[i]
        hours_val = hours_spent[i]
        inv_eff_val = inverter_eff[i]
        regen_val = regen[i]

        # Thermal Efficiency Delays
        temp_penalty = 1.0 + (max(0.0, t_val - 25.0) * 0.005) + (max(0.0, -5.0 - t_val) * 0.008)
        
        if engine_type[i] == 0:
            # Electric Engine Dynamics
            eff_factor = inv_eff_val / 100.0 if inv_eff_val > 0 else 0.95
            
            if p_wheel_val >= 0:
                p_traction_kw = p_wheel_val / eff_factor
            else:
                # Negative tractive power implies regenerative braking energy capture
                p_traction_kw = p_wheel_val * eff_factor * (regen_val / 100.0)
                
            hourly_kw = p_traction_kw + aux_val
            
            # Stationary safety check match
            if speed_val < 1.0:
                hourly_kw = aux_val
                
            hourly_kw = max(20.0, hourly_kw * temp_penalty)
            total_kwh = hourly_kw * hours_val
            
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
            hourly_kw = p_wheel_val + aux_val
            if speed_val < 1.0:
                hourly_kw = aux_val
                
            hourly_liters = max(5.0, (hourly_kw * temp_penalty) / (10.0 * 0.38))
            total_liters = hourly_liters * hours_val
            
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
    print("Dataset initialized inside rail_telemetry_data.csv with zero-mismatch math constants.")

if __name__ == "__main__":
    generate_telemetry_dataset()