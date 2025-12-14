import os
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl.mav_env import MavAltitudeEnv


timestamp = datetime.now().strftime("%d_%H-%M")
experiment_name = f"{timestamp}_PPO_altitude_hold"
base_dir = os.path.dirname(os.path.abspath(__file__))
exp_dir = os.path.join(base_dir, "experiments", experiment_name)
checkpoints_dir = os.path.join(exp_dir, "checkpoints")
tensorboard_dir = os.path.join(exp_dir, "logs")

os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

env = MavAltitudeEnv()

# Save a checkpoint every 100,000 steps or 10 episodes (1 episode = 2000 rl steps * 5 physics steps)
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=checkpoints_dir,
    name_prefix='model',
    save_replay_buffer=False
)

model = PPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log=tensorboard_dir, # Added for logging
    device='cuda' 
)

# --- Training ---
total_timesteps = 500_000
print(f"Training PPO for {total_timesteps} timesteps...")
print(f"\nTraining in: {exp_dir}\n")
model.learn(
    total_timesteps=total_timesteps,
    callback=checkpoint_callback,
    tb_log_name="training",
    progress_bar=True
)

final_model_path = os.path.join(exp_dir, "final_model")
model.save(final_model_path)
print("Training finished")
print(f"✓ Model saved: {final_model_path}.zip")