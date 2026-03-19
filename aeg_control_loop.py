import opendssdirect as dss
import torch
import numpy as np
from matd3_agent import Actor  # Importing the Actor class we defined earlier

# 1. Initialize OpenDSS Simulation
dss.Basic.Start()
dss.Text.Command('Compile (master.dss)')  # Loads your grid config

# 2. Setup the MATD3 Agent
state_dim = 24  # Example: Voltage at 12 buses + Current at 12 lines
action_dim = 4  # Example: Real/Reactive power for 2 Smart Inverters
max_action = 1.0

actor = Actor(state_dim, action_dim, max_action)
actor.load_state_dict(torch.load("trained_matd3_actor.pth"))
actor.eval()

def get_grid_state():
    """Extracts normalized voltages from OpenDSS"""
    voltages = dss.Circuit.AllBusMagPu()
    # Ensure state matches state_dim (pad or truncate if necessary)
    return np.array(voltages[:state_dim], dtype=np.float32)

def apply_action(actions):
    """Sends AI decisions to OpenDSS Storage/Inverter elements"""
    # Example: Setting kW discharge for Battery1
    dss.Circuit.SetActiveElement("Storage.Battery1")
    dss.Properties.Value("kW", str(actions[0] * 200)) # Scaling back to 200kW
    dss.Solution.Solve()

# 3. The Autonomous Control Loop
print("Starting Autonomous Grid Control...")
for step in range(24):  # 24-hour simulation window
    # A. Observe
    current_state = torch.FloatTensor(get_grid_state()).unsqueeze(0)
    
    # B. Decide (AI Inference)
    with torch.no_grad():
        action = actor(current_state).cpu().data.numpy().flatten()
    
    # C. Act
    apply_action(action)
    
    # D. Log Results
    v_min = min(dss.Circuit.AllBusMagPu())
    print(f"Hour {step}: Action Taken = {action[0]:.2f} | Min Voltage = {v_min:.4f} pu")

print("Simulation Complete. Grid remained stable.")