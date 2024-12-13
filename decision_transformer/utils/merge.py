import numpy as np
import torch
import copy

def merge_trajectories(old_traj: list, new_traj: list, reward_threshold=400, replace_in_place = False):
    filtered_new_traj = []
    if replace_in_place:
        for traj in new_traj:
            if traj['rewards'].sum().item() >= reward_threshold:
                filtered_new_traj.append(traj)
        
        old_traj_with_reward = []
        for traj in old_traj:
            old_traj_with_reward.append([traj['rewards'].sum().item(), traj])

        i = 0
        for traj_w_r in sorted(old_traj_with_reward):
            if i >= len(filtered_new_traj):
                break
            old_traj.remove(traj_w_r[1])
            i += 1
    else:
        filtered_new_traj = copy.deepcopy(new_traj)
    return old_traj + filtered_new_traj
    


