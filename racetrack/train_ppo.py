import gymnasium as gym
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

highway_env_name = ["highway-fast-v0", "intersection-v0", "racetrack-v0"]
index = 2

if index == 0:
    env_args = {"config":{
    "reward_speed_range": [24, 40],
    "high_speed_reward": 0.6
    }}
elif index == 1:
    env_args = {"config":{"action":{
    "type": "DiscreteMetaAction",
    "longitudinal": True,
    "lateral": False,
    "target_speeds": [0, 2.5, 5, 7.5, 10, 15]
    }}}
elif index == 2:
    # env_args = {"config":{"action":{
    # "type": "ContinuousAction",
    # "longitudinal": False,
    # "lateral": True
    # }}}
    # env_args = {"config":{"action":{
    # "type": "DiscreteAction",
    # "acceleration_range": [-5, 5],
    # "steering_range": [-0.7853981633974483, 0.7853981633974483],
    # "longitudinal": True,
    # "lateral": True,
    # "target_speeds": [0, 5, 10],
    # "lane_centering_reward": 2
    # }}}
    env_args = {"config":{
        "action":{
            "type": "ContinuousAction",
            "acceleration_range": [-2.5, 2.5],
            "steering_range": [-0.7853981633974483, 0.7853981633974483],
            "longitudinal": True,
            "lateral": True,
            "target_speeds": [0, 5, 10],
        },
        # "collision_reward": -2,
        # "lane_centering_reward": 1
        "duration": 210,
        "other_vehicles": 5,
        "show_trajectories": True,
        }
    }
else:
    raise NotImplementedError


vec_env = make_vec_env(highway_env_name[index], n_envs=1, env_kwargs=env_args)
# vec_env = make_vec_env(highway_env_name[index], n_envs=4)

model = PPO("MlpPolicy", 
            vec_env, 
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            gamma=0.9,
            verbose=1,
            tensorboard_log=f"highway_ppo/{highway_env_name[index]}")
model.learn(total_timesteps=6e4)
model.save(f"highway_ppo/ppo_model_{highway_env_name[index]}")

del model # remove to demonstrate saving and loading

model = PPO.load(f"highway_ppo/ppo_model_{highway_env_name[index]}")

obs = vec_env.reset()
# print(obs.shape)
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("rgb_array")