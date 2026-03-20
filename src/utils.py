import numpy as np
import pandas as pd
import time
import os

class ReplayBuffer:
    """
    Storage for MATD3 experience tuples (s, a, r, s', done).
    Essential for 'Centralized Training' of decentralized agents.
    """
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )

class GridLogger:
    """
    Handles CSV logging for generating the '35% Resilience' curves.
    """
    def __init__(self, filename="data/simulation_results/log.csv"):
        self.filename = filename
        self.data = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def log_step(self, step, v_min, v_max, reward, latency_ms):
        self.data.append({
            "step": step,
            "v_min_pu": v_min,
            "v_max_pu": v_max,
            "reward": reward,
            "latency_ms": latency_ms,
            "timestamp": time.time()
        })

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.filename, index=False)
        print(f"✔ Results saved to {self.filename}")

def compute_6G_latency(start_time):
    """
    Measures end-to-end inference latency to validate 6G URLLC claims.
    Returns time in milliseconds.
    """
    return (time.time() - start_time) * 1000

def normalize_voltage(v, v_base=1.0):
    """Scales voltage for NN input: (V - V_base) / 0.1"""
    return (v - v_base) / 0.1