import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

#adding parent directory to path so we can import 'model'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.mav_dynamics import MavDynamics
from tools.rotations import quaternion_to_euler

class MavAltitudeEnv(gym.Env):
    def __init__(self):
        super(MavAltitudeEnv, self).__init__()
        
        # 1. Physics Setup
        self.Ts = 0.01  # Physics time step
        self.mav = MavDynamics(self.Ts)
        
        # 2. Target
        self.target_altitude = 100.0
        
        # 3. Action Space: [Elevator, Throttle]
        # Elevator: -1 to 1 (Mapped to +/- 20 deg)
        # Throttle: 0 to 1 (Mapped to 0-100%)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )

        # 4. Observation Space: 
        # [Alt_Error, Climb_Rate, Pitch, Pitch_Rate, Airspeed]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.steps = 0
        self.max_steps = 2000  # 100 seconds per episode (if action_repeat=1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Reseting the physics object
        self.mav = MavDynamics(self.Ts)
        
        # State: [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        new_state = np.zeros((13, 1))
        init_speed = 35.0
        # 1. Position
        new_state[2] = -self.target_altitude # pd is negative altitude
        
        # 2. Velocity (Initialized u to 35 m/s)
        new_state[3] = init_speed # u
        
        # 3. Orientation (Quaternions for level flight are [1, 0, 0, 0])
        new_state[6] = 1.0 # e0
        
        self.mav._state = new_state
        self.mav._update_velocity_data(np.zeros((6,1)))
        # --- 2. TRIM STABILIZATION (NEW SECTION) ---
        # Applying a trim-like control signal for one step to prevent immediate stall/crash.
        # Estimated controls for straight and level flight at 35 m/s:
        # delta_e (Elevator): approx -0.2 rad (up-elevator)
        # delta_t (Throttle): approx 0.5 (65% power)
        initial_elevator = -0.2
        initial_throttle = 0.5
        initial_delta = np.array([0.0, initial_elevator, 0.0, initial_throttle]) 
    
        # Running one physics update with the initial trim control.
        # This stabilizes the forces/moments data, so the MavDynamics object starts near equilibrium.
        self.mav.update(initial_delta, wind=np.zeros((6,1)))
        return self._get_obs(), {}

    def step(self, action):

        # Elevator: Map [-1, 1] to approx [-20 deg, 20 deg]
        delta_e = np.clip(action[0], -1.0, 1.0) * np.deg2rad(10)
        
        # Throttle: 0 to 1
        delta_t = np.clip(action[1], 0.0, 1.0)
        
        # delta vector [aileron, elevator, rudder, throttle]
        delta = np.array([0.0, delta_e, 0.0, delta_t])

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

    
        alt_error = self.target_altitude - self.mav.get_altitude()
        va = self.mav._Va # Current airspeed
        MIN_VA = 15.0
        # Airspeed Penalty (NEW: Force the agent to avoid stall)
        # Penalize if airspeed is below the safe minimum.
        if abs(alt_error) < 5.0:
            reward += -0.5 * (alt_error**2)
        elif abs(alt_error) < 20.0:
            reward += -2.0*(alt_error**2)
        else:
            reward += -5.0 * (alt_error**2) 
        if va < MIN_VA:
            reward += -10.0*(MIN_VA - va)
        reward += -0.01 * delta_t 
        reward += -0.01 * (delta_e**2) # Penalize wild elevator flapping
        
        # Survival Bonus
        reward += 1.0
        # attitude check
        phi, theta, psi = quaternion_to_euler(self.mav._state[6:10])
        if abs(theta) > np.deg2rad(60):
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

        # Climb rate approx = -w (in body frame) or use pd_dot
        # Let's use pd_dot (vertical velocity)
        # We can re-calculate it or just use -w for approximation near level flight
        w = self.mav._state.item(5)
        
        obs = np.array([
            self.target_altitude - h,
            -w,        # Approx climb rate
            theta,
            q,
            va
        ], dtype=np.float32)
        
        return obs