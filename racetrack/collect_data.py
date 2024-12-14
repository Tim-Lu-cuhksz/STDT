import gymnasium as gym
import highway_env
from tqdm import tqdm

from stable_baselines3 import PPO
from minari import DataCollector

env_id = "racetrack-v0"
total_episodes = int(256)

env = gym.make(env_id, render_mode="rgb_array", config={"action":{
    "type": "ContinuousAction",
    "acceleration_range": [-5, 5],
    "steering_range": [-0.7853981633974483, 0.7853981633974483],
    "longitudinal": False,
    "lateral": True,
    "target_speeds": [0, 5, 10]},
    "other_vehicles": 5,
    "show_trajectories": True,
    "duration": 210}
)
env = DataCollector(env, record_infos=True)
print(env.action_space)

# model_name = f"highway_ppo/ppo_model_{env_id}_continuous_lateral_envs_1_step_40000"
model_name = f"highway_ppo/ppo_model_{env_id}"

model = PPO.load(model_name)
for _ in tqdm(range(total_episodes)):
    done = truncated = False
#   obs, info = vec_env.reset()
    obs, info = env.reset()
    # print(obs.shape)
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        # print(action)
        # obs, reward, done, truncated, info = vec_env.step(action)
        obs, reward, done, truncated, info = env.step(action)
        # env.render()

dataset = env.create_dataset(
    dataset_id="racetrack/test-v2",
)