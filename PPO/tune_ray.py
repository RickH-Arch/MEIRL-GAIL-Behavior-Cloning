import os
import sys
sys.path.append(os.getcwd())

from PPO.env import RegionSensor

import ray
from ray import air,tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.tune import ResultGrid

from grid_world.envGen_grid_world import GridWorld_envGen
from ray.rllib.env.env_context import EnvContext
import gymnasium as gym
import numpy as np


torch, nn = try_import_torch()



if ray.is_initialized(): ray.shutdown()
ray.init(local_mode = True,include_dashboard=True,ignore_reinit_error=True)
print('----->',ray.get_gpu_ids())
print('----->',torch.cuda.is_available())
print('----->',torch.cuda.device_count())

storage_path = os.getcwd()+"/ray_result"
exp_name = "ppo_demo"

config = (
    get_trainable_cls('PPO')
    .get_default_config()
    .environment(RegionSensor,env_config = {
        "width":10,
        "height":10,
        'envs_img_folder_path': os.getcwd()+'/demo_dmeirl/demo_label/train',
        'target_svf_delta':{50:0.5,40:1,30:1,20:1,10:1,0:1},
        'model_path':os.getcwd()+'/demo_dmeirl/demo_result/1_model.pth',
        'max_step_count':20,
        'experts_traj_path':os.getcwd()+'/demo_dmeirl/demo_expert_trajs_0205.csv'},
    )
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "1")))
)
#config.environment(disable_env_checking=True)

stop = {
    'training_iteration': 10
}

print("Training automatically with tune stopped after {} iterations".format(stop['training_iteration']))

tuner = tune.Tuner(
    'PPO',
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        name = exp_name,
        stop=stop,
        storage_path=storage_path,
    )
)

result_grid : ResultGrid = tuner.fit()

ray.shutdown()
