# Self-Trained Decision Transformer

This is the repository for Self Trained Decision Transformer (STDT), a course project at University of Waterloo. Refer to [link](#installation) for packages installation and run an example [here](#example). The report could be found in the [link](ECE_750_T40_Final_Report.pdf).

## Acknowledgment
We acknowledge the work of [Decision Transformer](https://github.com/kzl/decision-transformer) upon which much of the codes in this repository are built.

## Video Demonstrations of STDT
### Lane Following
![lane_follow_demo](Highway-env-lane-tracking.gif)


### Overtaking
![overtaking_demo](Highway-env-success.gif)

### Failed Overtaking
![failed_overtaking_demo](Highway-env-overtaking-fail.gif)

## Installation
Please install [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv), and [Minari](https://github.com/Farama-Foundation/Minari) according to their instructions.

## Example
You can execute the following codes to collect Offine dataset and self-train a DT.
### Offline Dataset Collection
We use Proximal Policy Optimization (PPO) as provided in [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) to train our Offline agent. You could train the agent using the following code:
```
python racetrack/train_ppo.py
```
We could then use the PPO policy to generate our Offline dataset (via [Minari](https://github.com/Farama-Foundation/Minari)) by executing the code below:
```
python racetrack/collect_data.py
```
The dataset we collected is named as ```racetrack/test-v2``` and is stored locally. To load the dataset, run the code:
```
import minari

dataset = minari.load_dataset("racetrack/test-v2")
```

### Self-Training a DT
We could use the following code to test STDT:
```
python experiment.py --num_eval_episodes 10 --max_iters 4 --num_steps_per_iter 500 --self_train True --num_gen_episodes 15 --num_episode_updates 5 --log_to_wandb True
```
Note that you should get your [wandb](https://wandb.ai/site) account ready to see the results.

You can also train and evaluate DT only by setting ```--self_train False --num_episode_updates 1```.
