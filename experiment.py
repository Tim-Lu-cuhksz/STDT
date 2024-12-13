# import gym
import gymnasium as gym
from gymnasium import spaces
import highway_env
import numpy as np
import torch
import wandb
import minari

import argparse
import pickle
import random
import sys
from tqdm import tqdm

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, generate_episodes
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.utils.merge import merge_trajectories

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
        datasets = None,
        trainer_old = None
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif env_name == 'racetrack':
        env = gym.make(f'{env_name}-v0', render_mode="rgb_array", config={"action":{
        "type": "ContinuousAction",
        "acceleration_range": [-5, 5],
        "steering_range": [-0.7853981633974483, 0.7853981633974483],
        "longitudinal": False,
        "lateral": True,
        "target_speeds": [0, 5, 10]},
        "duration": 10,
        "other_vehicles": 5,
        "show_trajectories": True}
        )
        max_ep_len = 1502
        env_targets = [1000, 1500]
        scale = 1.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = spaces.utils.flatdim(env.observation_space)
    # print(state_dim)
    # state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    if env_name == "racetrack":
        processed = True
        if datasets is None:
            trajectories_raw = minari.load_dataset(f'{env_name}/{dataset}')
            trajectories = []
            for traj_raw in trajectories_raw:
                traj_dict = dict()
                num_steps = traj_raw.actions.shape[0]
                obs_raw = torch.tensor(traj_raw.observations).flatten(start_dim=1)
                # print(obs_raw.shape)
                traj_dict['observations'] = obs_raw.cpu().numpy()[:num_steps, : ]
                traj_dict['next_observations'] = obs_raw.cpu().numpy()[1:, : ]
                traj_dict['terminals'] = traj_raw.terminations | traj_raw.truncations
                traj_dict['actions'] = traj_raw.actions
                traj_dict['rewards'] = traj_raw.rewards
                trajectories.append(traj_dict)
        else:
            trajectories = datasets
    else:
        dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if env_name == "racetrack" and not processed:
            states.append(path.observations)
            traj_lens.append(len(path.observations))
            returns.append(path.rewards.sum())
        else:
            if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            if not isinstance(path, dict):
                print(path[0])
                print(type(path[0]))
                print(len(path))
                raise TypeError
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    states_flatten = torch.tensor(states).flatten(start_dim=1)
    # print(states.shape)
    state_mean, state_std = np.mean(states_flatten.cpu().numpy(), axis=0), np.std(states_flatten.cpu().numpy(), axis=0) + 1e-6
    # state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            if env_name == "racetrack" and not processed:
                si = random.randint(0, traj.rewards.shape[0] - 1)

                # get sequences from dataset
                state_ = torch.tensor(traj.observations[si:si + max_len]).flatten(start_dim=1)
                s.append(state_.cpu().numpy().reshape(1, -1, state_dim))
                # state_ = state_.cpu().numpy()
                # print(state_.cpu().numpy().reshape(1, -1, state_dim).shape)
                a.append(traj.actions[si:si + max_len].reshape(1, -1, act_dim))
                # print(traj.actions[si:si + max_len].shape)
                # print(traj.actions[si:si + max_len].reshape(1, -1, act_dim).shape)
                r.append(traj.rewards[si:si + max_len].reshape(1, -1, 1))
                d.append(traj.terminations[si:si + max_len].reshape(1, -1))

                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
                rtg.append(discount_cumsum(traj.rewards[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            else:
                si = random.randint(0, traj['rewards'].shape[0] - 1)

                # get sequences from dataset
                s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
                a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
                r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                if 'terminals' in traj:
                    d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
                # elif 'terminations' in traj
                else:
                    d.append(traj['dones'][si:si + max_len].reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
                rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))

            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = tlen_ = s[-1].shape[1]
            tlen_ = a[-1].shape[1]
            # if tlen != a[-1].shape[1]:
            #     print("n_state:", tlen)
            #     print("n_actions:", a[-1].shape[1])
            # assert(tlen == a[-1].shape[1])
            # print(tlen)
            # print(s[-1].shape)
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen_, act_dim)) * -10., a[-1]], axis=1)
            # print(a[-1].shape)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen_, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen_)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen_, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen_)), np.ones((1, tlen_))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        # print(len(a))
        # print('='*10)
        # for i in a:
        #     print(i.shape)
        # print('='*10)

        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        # print(a.shape)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if trainer_old is not None:
        trainer = trainer_old
    else:
        if model_type == 'dt':
            trainer = SequenceTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )
        elif model_type == 'bc':
            trainer = ActTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )

    if log_to_wandb:
        wandb.init(
            name=f'{exp_prefix}-self-train-test',
            # name=f'{exp_prefix}',
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in tqdm(range(variant['max_iters'])):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

    self_train = variant['self_train']
    if self_train:
        new_trajectories = []
        print('Start generating new episodes')
        num_gen_episodes = variant['num_gen_episodes']
        for i in range(num_gen_episodes):
            traj_dict_new = {}
            ret, length, states_gen, actions_gen, rewards_gen = generate_episodes(
                env,
                state_dim,
                act_dim,
                model,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=env_targets[1]/scale,
                mode=mode,
                state_mean=state_mean,
                state_std=state_std,
                device=device,
            )
            traj_len = actions_gen.shape[0]
            # print(states_gen.shape[0])
            # print(actions_gen.shape[0])
            # assert states_gen.shape[0] == actions_gen.shape[0]
            traj_dict_new['observations'] = states_gen[ : traj_len, : ]
            traj_dict_new['next_observations'] = states_gen[ 1 : , : ]
            traj_dict_new['actions'] = actions_gen
            traj_dict_new['rewards'] = rewards_gen
            terminals = torch.zeros(actions_gen.shape[0])
            terminals[-1] = 1
            traj_dict_new['terminals'] = terminals
            new_trajectories.append(traj_dict_new)
        # Select num_gen_episodes episodes with lowest returns from trajectories and remove them
        # Add the new trajectories to the old ones
        # print('merging two trajectories...')
        assert isinstance(trajectories, list)
        assert isinstance(new_trajectories, list)
        assert isinstance(new_trajectories[0], dict)
        return merge_trajectories(trajectories, new_trajectories), trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='racetrack')
    parser.add_argument('--dataset', type=str, default='test-v2')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000) # 1e4
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--num_steps_per_iter', type=int, default=10) # 1e4
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--self_train', type=bool, default=True)
    parser.add_argument('--num_gen_episodes', type=int, default=5)
    
    args = parser.parse_args()

    num_traj_updates = 5
    trajectories = None
    for i in range(num_traj_updates):
        if i == 0:
            trajectories, trainer = experiment('gym-experiment', variant=vars(args))
            continue
        trajectories, trainer = experiment('gym-experiment', variant=vars(args), datasets=trajectories, trainer_old=trainer)
