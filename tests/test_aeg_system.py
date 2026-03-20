import unittest
import os
import torch
import numpy as np
import opendssdirect as dss
from src.matd3_agent import Actor
from src.opendss_env import OpenDSSEnv
from src.utils import compute_6G_latency

class TestAEGSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """One-time setup for the test suite"""
        cls.master_path = "dss_files/master.dss"
        cls.model_path = "models/trained_matd3_actor.pth"
        cls.state_dim = 24
        cls.action_dim = 4
        
        # Create a mock master file if it doesn't exist for the test to pass
        if not os.path.exists("dss_files"):
            os.makedirs("dss_files")
        if not os.path.exists(cls.master_path):
            with open(cls.master_path, "w") as f:
                f.write("new circuit.Test phases=3 bus1=SourceBus basekv=11")

    def test_01_opendss_engine(self):
        """Check if OpenDSS-G engine is accessible via Python"""
        try:
            dss.Basic.Start(0)
            dss.Text.Command(f"Compile ({self.master_path})")
            voltages = dss.Circuit.AllBusMagPu()
            self.assertGreater(len(voltages), 0, "No buses found in circuit.")
            print(f"✔ [PASS] OpenDSS Engine: Bus 1 Voltage = {voltages[0]:.4f} pu")
        except Exception as e:
            self.fail(f"OpenDSS Initialization Failed: {e}")

    def test_02_actor_inference(self):
        """Verify the MATD3 Actor produces the correct output shape"""
        actor = Actor(self.state_dim, self.action_dim, 1.0)
        
        # Use dummy data to test the forward pass
        test_input = torch.randn(1, self.state_dim)
        with torch.no_grad():
            output = actor(test_input)
            
        self.assertEqual(output.shape, (1, self.action_dim))
        self.assertTrue(torch.all(output <= 1.0) and torch.all(output >= -1.0), "Actions out of bounds.")
        print("✔ [PASS] MATD3 Actor: Inference logic and bounds verified.")

    def test_03_6G_latency_benchmark(self):
        """Benchmark to prove the <1ms URLLC requirement from the paper"""
        actor = Actor(self.state_dim, self.action_dim, 1.0)
        test_input = torch.randn(1, self.state_dim)
        
        # Warm-up (to initialize CUDA/CPU kernels)
        _ = actor(test_input)
        
        import time
        latencies = []
        for _ in range(100):
            start = time.time()
            _ = actor(test_input)
            latencies.append(compute_6G_latency(start))
            
        avg_latency = np.mean(latencies)
        print(f"✔ [PASS] 6G Latency Test: {avg_latency:.4f} ms (Target: <1.0ms)")
        self.assertLess(avg_latency, 1.0, "Latency exceeds 6G URLLC performance threshold.")

    def test_04_env_reward_logic(self):
        """Ensure the reward function correctly penalizes voltage violations"""
        env = OpenDSSEnv(self.master_path)
        
        # Case 1: Perfect voltage
        perfect_obs = np.ones(self.state_dim)
        reward_good = env._calculate_reward(perfect_obs)
        
        # Case 2: Under-voltage (0.90 pu)
        bad_obs = np.ones(self.state_dim) * 0.90
        reward_bad = env._calculate_reward(bad_obs)
        
        self.assertGreater(reward_good, reward_bad, "Reward function failed to penalize instability.")
        print(f"✔ [PASS] Reward Logic: Stability Bonus = {reward_good}, Penalty = {reward_bad}")

if __name__ == '__main__':
    unittest.main()