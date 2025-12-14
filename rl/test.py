import os
import sys
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from rl.mav_env import MavAltitudeEnv

# --- Setup Paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))
# Option 1: Use the latest final model 
latest_exp_dir = os.path.join(base_dir, "experiments")
experiments = sorted([d for d in os.listdir(latest_exp_dir) if os.path.isdir(os.path.join(latest_exp_dir, d))])
if experiments:
    latest_exp = experiments[-1]  
    model_path = os.path.join(latest_exp_dir, latest_exp, "final_model.zip")
else:
    model_path = None
    print("No experiments found!")
    sys.exit()

# Option 2: Use specific checkpoint 
# model_path = os.path.join(base_dir, "experiments", "15_00-30_PPO_altitude_hold", "checkpoints", "model_100000_steps.zip")
try:
    env = MavAltitudeEnv()
    model = PPO.load(model_path, env=env)
    print(f"Successfully loaded model from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have run train_agent.py first to create the model file.")
    sys.exit()

MAX_EPISODE_LENGTH = 1000 
obs, _ = env.reset()

# Data storage
time_hist = []
alt_hist = []
alt_target_hist = []
throttle_hist = []

t = 0.0
for i in range(MAX_EPISODE_LENGTH):

    action, _ = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    time_hist.append(t)
    alt_hist.append(env.mav.get_altitude())
    alt_target_hist.append(env.target_altitude)
    throttle_hist.append(action[1])
    
    t += env.Ts * 5 # Ts * sim_steps_per_action = 0.01 * 5 = 0.05 seconds per RL step
    
    if terminated or truncated:
        print(f"Episode terminated at time {t:.2f}s with reward {reward:.2f}.")
        break

# Converting to numpy arrays for plotting
time_hist = np.array(time_hist)
alt_hist = np.array(alt_hist)
throttle_hist = np.array(throttle_hist)

plt.figure(figsize=(12, 8))

# Subplot 1: Altitude Response
plt.subplot(2, 1, 1)
plt.plot(time_hist, alt_hist, label="Actual Altitude (h)")
plt.plot(time_hist, alt_target_hist, 'r--', label=f"Target Altitude ({env.target_altitude}m)")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("RL Agent Altitude Hold Performance")
plt.grid(True)
plt.legend()
plt.ylim(env.target_altitude - 20, env.target_altitude + 20) # Focus zoom on target

# Subplot 2: Throttle Command (Energy Usage)
plt.subplot(2, 1, 2)
plt.plot(time_hist, throttle_hist, 'g-', label="Throttle Command ($\delta_t$)")
plt.xlabel("Time (s)")
plt.ylabel("Throttle (0 to 1)")
plt.title("Throttle Command (Energy Usage) Over Time")
plt.grid(True)
plt.legend()
throttle_min = throttle_hist.min()
throttle_max = throttle_hist.max()
throttle_range = throttle_max - throttle_min

if throttle_range < 0.1:  # If variation is very small
    # Zoom in around the actual values with 20% margin
    margin = 0.05
    plt.ylim(throttle_min - margin, throttle_max + margin)
    print(f"Throttle variation detected: {throttle_range:.4f} (zoomed in)")
else:
    # Show full range if there's significant variation
    plt.ylim(0, 1.1)

plt.tight_layout()
plt.show()