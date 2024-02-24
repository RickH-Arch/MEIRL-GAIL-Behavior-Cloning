import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.rllib.utils.framework import try_import_torch
import os
print(os.getcwd())


import sys
sys.path.append(os.getcwd())

from PPO.env import RegionSensor


import gymnasium as gym
import numpy as np
import torch

if ray.is_initialized(): ray.shutdown()
ray.init(num_cpus=1, num_gpus=1,include_dashboard=True,ignore_reinit_error=True,)
print('----->',ray.get_gpu_ids())
print('----->',torch.cuda.is_available())
print('----->',torch.cuda.device_count())


config = (
    PPOConfig().environment(
        env=RegionSensor,
        env_config={"width":10,
        "height":10,
        'envs_img_folder_path': os.getcwd()+'/demo_dmeirl/demo_label/train',
        'target_svf_delta':{50:0.5,40:1,30:1,20:1,10:1,0:1},
        'model_path':os.getcwd()+'/demo_dmeirl/demo_result/1_model.pth',
        'max_step_count':20,
        'experts_traj_path':os.getcwd()+'/demo_dmeirl/demo_expert_trajs_0205.csv'},
    )
    .framework("torch")
    .rollouts(num_rollout_workers=1)
)

algo = config.build()

for i in range(2):
    results = algo.train()
    print(f"========Iter: {i}; avg.reward={results['episode_reward_mean']}========")