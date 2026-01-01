import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.mav_dynamics import MavDynamics
from tools.rotations import quaternion_to_euler

class MavAltitudeEnv(gym.Env):
    def __init__(self):
        super(MavAltitudeEnv, self).__init__()
        
        # 1. Physics Setup
        self.Ts = 0.01  
        self.mav = MavDynamics(self.Ts)
        
        # 2. Action Space: [Elevator, Throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

        # 4. Observation Space: 
        # [Alt_Error, Integral_Alt_Error, climb rate, Pitch, Pitch_Rate, airspeed]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -0.1, -0.1, -np.inf]), 
            high=np.array([np.inf, np.inf, np.inf, 0.1, 0.1, np.inf]), 
            dtype=np.float32
        )
        
        self.int_error = 0
        self.target_altitude = 100.0
        self.prev_throttle = 0.5
        self.steps = 0
        self.max_steps = 2000  # 100 seconds per episode (if action_repeat=1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.int_error = 0.0
        self.mav = MavDynamics(self.Ts)
        self.prev_throttle = 0.5
    
        new_state = np.zeros((13, 1)) # State: [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        init_speed = 25.0
        # 1. Position
        new_state[2] = -self.target_altitude # pd is negative altitude
        
        # 2. Velocity (Initialized u to 35 m/s)
        new_state[3] = init_speed # u
        
        # 3. Orientation (Quaternions for level flight are [1, 0, 0, 0])
        new_state[6] = 1.0 # e0
        
        self.mav._state = new_state
        self.mav._update_velocity_data(np.zeros((6,1)))
        # --- 2. TRIM STABILIZATION (NEW SECTION) ---
        initial_elevator = -0.2
        initial_throttle = 0.5
        initial_delta = np.array([0.0, initial_elevator, 0.0, initial_throttle]) 
    
        # Running one physics update with the initial trim control.
        # This stabilizes the forces/moments data, so the MavDynamics object starts near equilibrium.
        self.mav.update(initial_delta, wind=np.zeros((6,1)))
        return self._get_obs(), {}

    def step(self, action):

        delta_e = np.clip(action[0], -1.0, 1.0) * np.deg2rad(20) # Elevator: Map [-1, 1] to approx [-20 deg, 20 deg]
        delta_t = np.clip(action[1], 0.0, 1.0)                   # Throttle: 0 to 1
        delta = np.array([0.0, delta_e, 0.0, delta_t])           # delta vector [aileron, elevator, rudder, throttle]
        # Run Physics
        # We run 5 physics steps for every 1 RL step to speed up training
        # (Control @ 20Hz, Physics @ 100Hz)
        reward = 0
        terminated = False
        for _ in range(5): 
            self.mav.update(delta, wind=np.zeros((6,1)))
            # Crash Check (Ground collision)
            if self.mav.get_altitude() <= 0:
                terminated = True
                reward = -500.0 
                break
        if terminated:
            return self._get_obs(), reward, terminated, False, {}

    
        alt_error = self.target_altitude - self.mav.get_altitude()
        q = self.mav._state.item(11)
        va = self.mav._Va # Current airspeed

        # Boundary Check: 
        if abs(alt_error) > 100.0:
            terminated = True
            reward = -500.0 
            return self._get_obs(), reward, terminated, False, {}
        
        # Integral update (simple rectangle rule)
        self.int_error += alt_error * (self.Ts * 5.0)  # 5 physics steps
        self.int_error = np.clip(self.int_error, -500.0, 500.0)

        # Reward for altitude error (Gaussian)
        reward += 2.0 * np.exp(-(alt_error**2) / (2 * 5.0**2))

        # Airspeed Penalty (NEW: Force the agent to avoid stall)
        MIN_VA = 15.0
        if va < MIN_VA:
            reward += -2.0*(MIN_VA - va)**2    # Penalize if airspeed is below the safe minimum.

        reward += -0.05 * (q**2)               # smooth pitch rate
        reward += -0.01 * (delta_e**2)         # Penalize wild elevator flapping
        throttle_delta = abs(delta_t - self.prev_throttle)
        reward -= 0.5 * throttle_delta         # Penalize large jumps
        self.prev_throttle = delta_t

        # Survival Bonus
        reward += 2.0
        _, theta, _ = quaternion_to_euler(self.mav._state[6:10])
        reward -= 0.5 * (theta**2)

        if abs(theta) > np.deg2rad(45):
            terminated = True
            reward += -250.0
            
        # Checking Truncation (Time limit)
        self.steps += 1
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        
        h = self.mav.get_altitude()
        va = self.mav._Va

        q_idx = 11
        q = self.mav._state.item(q_idx)
        
        e0, e1, e2, e3 = self.mav._state[6:10].flatten()
        
        sin_theta = 2.0 * (e0*e2 - e1*e3)
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        theta = np.arcsin(sin_theta)

        w = self.mav._state.item(5)
        
        obs = np.array([
            (self.target_altitude - h) / 50.0,
            self.int_error / 500.0,
            -w / 10.0,        # Approx climb rate
            theta / np.deg2rad(45),
            q,
            (va - 25.0) /10.0
            
        ], dtype=np.float32)
        
        return np.clip(obs, -2.0, 2.0)