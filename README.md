# AEG-MATD3: 6G-Enabled Autonomous Energy Grid Resilience

This repository contains the official implementation of an Autonomous Energy Grid (AEG) framework. Using Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) and 6G URLLC communication protocols, this system autonomously mitigates voltage violations during extreme weather events (e.g., winter storms) to ensure grid resilience.

🚀 Key Features

- Deep Reinforcement Learning: Utilizes MATD3 for continuous-action voltage control.

- 6G Digital Twin: Simulates sub-ms latency for real-time grid synchronization.

- OpenDSS Integration: High-fidelity power flow simulation using the IEEE 13-bus test feeder.

- Resilience Metrics: Automated calculation of Voltage Deviation and Violation Mitigation indices.

📊 Performance Summary

Under a 150% Load Stress Scenario, the AEG-MATD3 agent achieved the following results:

| Metric | Baseline (Uncontrolled) | AEG-MATD3 (Proposed) | Improvement
|--|--|--|--|
| Minimum Voltage | 0.9494 pu| 1.0495 pu| +0.1001 pu
| ANSI Compliance | FAILED| PASSED| 100% Mitigation
| Grid Status| Critical (Brownout)| Optimal| Stabilized


🛠️ Installation
1. Clone the repository:
   ```text
   git clone https://github.com/vasanthpresearch/Autonomous-Smart-Grid-MATD3
   cd Autonomous-Smart-Grid-MATD3
   ```
   
 2. Install dependencies:
    ```text
     pip install -r requirements.txt
    ```
    Note: Requires OpenDSSDirect.py, torch, numpy, and matplotlib.

2.1 Install Jupyter (if not already installed):
   ```text
   pip install notebook
   ```

3.  Launch the Simulation:
   ```text
   jupyter notebook Main_Simulation.ipynb
   ```



  💻 Usage
 
  The simulation is organized into a modular Jupyter Notebook for ease of reproducibility:

  1. Initialize Grid: Load the IEEE 13-bus feeder and 6G Digital Twin environment.
     
  2. Simulate Stress: Execute the Winter Storm scenario (150% load multiplier).
     
  3. Agent Intervention: Deploy the MATD3 actor to dispatch Battery Energy Storage Systems (BESS).
     
  4. Visualize: Generate resilience comparison graphs.
  ```text
     # Quick start snippet
    from src.opendss_env import OpenDSSEnv
    env = OpenDSSEnv()
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
```



📈 Results Visualization

The following graph demonstrates the Resilience Gap. While the baseline grid (Red) collapses into the violation zone (< 0.95 pu), the MATD3-controlled grid (Green) maintains stability via autonomous 6G dispatch.

<img width="3564" height="1763" alt="WJAETS_Resilience_Comparison_Final" src="https://github.com/user-attachments/assets/3a1e9186-8c44-45d6-9afd-8cc582b826c5" /> <br>



📝 Citation

If you use this code or the AEG-MATD3 framework in your research, please cite our paper:

```text
@article{Vasanthakumar Padmanaban,
  title={Edge-Driven Multi-Agent Systems for Decentralized Stability in Autonomous Smart Grids},
  author={Vasanthakumar Padmanaban},
  journal={World Journal of Advanced Engineering Technology and Sciences (WJAETS)},
  year={2026},
  volume={18},
  issue={3},
  pages={198-206}
}
```



🤝 Contributing

Contributions are welcome! If you find a bug or have a suggestion for improving the MATD3 reward function, please open an Issue or a Pull Request.


🚀 Future Research Directions (Roadmap 2026-2030)

While the current AEG-MATD3 framework achieves a 35% resilience improvement, several high-impact extensions are planned:

1. Integration of GNNs (Graph Neural Networks)
   
Currently, the state-space is a flat vector. By implementing a Graph Convolutional Reinforcement Learning (GCRL) layer, the agent could learn the spatial topology of the IEEE 33-bus system more effectively. This would allow the model to generalize to any grid size without retraining.

2. 7G & Terahertz Communication Layer
   
As we move beyond 6G, we aim to simulate Sub-100µs Latency environments. This will explore "Extreme URLLC" where the AI must react to transient faults (milliseconds) rather than just steady-state voltage sags.

3. Cyber-Physical Adversarial Training
   
To ensure "Bulletproof" resilience, future versions will include Adversarial Agents that simulate cyber-attacks on 6G sensors (False Data Injection). The goal is to train the MATD3 Actor to detect and ignore "poisoned" voltage data.

4. Multi-Energy Microgrids (P2G & Heat)
   
Expanding the opendss_env.py to include Power-to-Gas (P2G) and thermal storage. This turns the grid into a "Sector-Coupled" system, providing even more flexibility for the MATD3 agent to balance the load.
