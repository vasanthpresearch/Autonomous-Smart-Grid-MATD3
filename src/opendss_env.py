

import os
import opendssdirect as dss
import numpy as np
import pandas as pd

class OpenDSSEnv:
    def __init__(self, master_file):
        # 1. Path Safety: Ensures OpenDSS finds the .dss files regardless of where you run the script
        self.master_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', master_file))
        self.load_circuit()
        
        # 1.1 Load the stress factor from our CSV data
        load_profiles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'bus_data', 'load_profiles.csv'))
        load_data = pd.read_csv(load_profiles_path)

        # 2. Dynamic State Mapping: Automatically detects how many buses are in the grid
        v_pu = dss.Circuit.AllBusMagPu()
        self.state_dim = len(v_pu) 
        
        # 3. Load Memory: Store original kW values to allow for the '35% Stress' scaling
        self.base_loads = {}
        load_names = dss.Loads.AllNames()
        for name in load_names:
            dss.Loads.Name(name)
            self.base_loads[name] = dss.Loads.kW() 
        
        # 4. Agent Configuration
        self.action_dim = 4    # P/Q control for 2 Smart Inverters/Batteries
        self.v_min = 0.95      # ANSI C84.1 Standards
        self.v_max = 1.05
        
        print(f"✔ Environment Initialized with {self.state_dim} buses and {len(self.base_loads)} loads.")

    def load_circuit(self):
        """Compiles the DSS circuit and initializes the engine"""
        dss.Basic.Start(0)
        dss.Text.Command(f"Compile ({self.master_file})")
        dss.Solution.Solve()

    def reset(self):
        """Resets the grid to the initial state for a new simulation episode"""
        self.load_circuit()
        return self._get_obs()

    # Removed invalid commented-out _get_obs method that caused syntax error

    def _get_obs(self):
        v_pu = dss.Circuit.AllBusMagPu()
        obs = np.array(v_pu[:self.state_dim], dtype=np.float32)
        
        # 2026 Resilience Check: Alert if scaling is wrong
        if np.any(obs > 2.0):
            print("⚠️ WARNING: Voltage scaling error detected. Check 'CalcVoltageBases' in master.dss")
            return obs / 6350.85 # Automatic emergency scaling
            
        return obs  

    def step(self, actions, hour_of_day):
        """
        actions: Normalized [-1, 1] from MATD3
        hour_of_day: 0-23 (to index the load profile)
        """
        # Load the stress factor from our CSV data
        load_profiles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'bus_data', 'load_profiles.csv'))
        load_data = pd.read_csv(load_profiles_path)
        # Forward-fill multiplier for missing hours
        if hour_of_day not in load_data['hour'].values:
            prev_hours = load_data[load_data['hour'] <= hour_of_day]
            if not prev_hours.empty:
                multiplier = prev_hours.iloc[-1]['load_multiplier']
            else:
                multiplier = load_data.iloc[0]['load_multiplier']  # fallback to first
        else:
            multiplier = load_data.loc[load_data['hour'] == hour_of_day, 'load_multiplier'].values[0]

        # 2. Apply this stress to ALL loads in OpenDSS
        # This simulates the "Peak Demand" that causes the voltage to drop
        for name, base_kw in self.base_loads.items():
            dss.Loads.Name(name)
            dss.Loads.kW(base_kw * multiplier)

        # 3. Apply AI Action (Battery Dispatch)
        # Map [-1, 1] to [-500kW, 500kW]
        p_gen = actions[0] * 500 
        dss.Circuit.SetActiveElement("Storage.Battery1")
        dss.Properties.Value("kW", str(p_gen))

        # 4. Solve and get New State
        dss.Solution.Solve()
        obs = self._get_obs()
        
        # 5. Calculate Resilience Reward
        reward = self._calculate_reward(obs)
        
        return obs, reward, False

    def _calculate_reward(self, obs):
        """
        Reward function designed for 35% resilience:
        Penalizes voltage deviation from 1.0 pu and rewards staying in bounds.
        """
        # Quadratic penalty for voltage deviation
        deviation = np.sum((obs - 1.0)**2)
        
        # Bonus for staying within ANSI limits [0.95, 1.05]
        stability_bonus = 1.0 if (np.min(obs) >= self.v_min and np.max(obs) <= self.v_max) else -5.0
        
        return float(stability_bonus - (10 * deviation))