import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray.rllib.utils.framework import try_import_torch
import os
print(os.getcwd())


import sys
sys.path.append(os.getcwd())

#from env import RegionSensor

from grid_world.envGen_grid_world import GridWorld_envGen
from ray.rllib.env.env_context import EnvContext
import gymnasium as gym
import numpy as np
import torch



if ray.is_initialized(): ray.shutdown()
ray.init(include_dashboard=True,ignore_reinit_error=True,
         num_gpus=1)
print('----->',ray.get_gpu_ids())
print('----->',torch.cuda.is_available())
print('----->',torch.cuda.device_count())

class RegionSensor(gym.Env):
    def __init__(self,config:EnvContext):
        '''
        config: width,height,envs_img_folder_path,
        target_svf_delta: dict, key:state_active, value:delta,
        model_path: path of model that used by env gen world to convert feature to reward,
        max_step_count: 
        '''
        self.width = config['width']
        self.height = config['height']
        self.envs_img_folder_path = config['envs_img_folder_path']
        self.target_svf_delta = config['target_svf_delta']
        self.model_path = config['model_path']
        self.max_step_count = config['max_step_count']
        self.experts_traj_path = config['experts_traj_path']

        self.world = GridWorld_envGen(self.width,self.height,
                                       self.envs_img_folder_path,
                                       self.experts_traj_path,
                                       self.target_svf_delta,
                                       self.model_path)
        
        self.cur_env_np = self.origin_env_np = np.array(self.world.parser.environments_arr)
        self.init_state = self.origin_env_np.reshape(-1)
        self.cur_state = self.init_state
        self.action_space = gym.spaces.MultiDiscrete([self.origin_env_np.shape[0],self.origin_env_np.shape[1],self.origin_env_np.shape[2],2])
        self.observation_space = gym.spaces.Box(low=0.,high=1.0,shape=(len(self.init_state),),dtype=np.int32)
        self.step_count = 0

    def reset(self,*,seed=None,options=None):
        self.cur_state = self.init_state
        self.cur_env_np = self.origin_env_np
        self.step_count = 0
        return np.array(self.cur_state,dtype=np.int32),{}
    
    def step(self,action):
        '''
        action: [feature_idx,state,status]
        if state == 0, then the feature is set to 0
        if state == 1, then the feature is set to 1
        '''
        #parse action
        feature_idx = action[0]
        state_y = action[1]
        state_x = action[2]
        status = action[3]

        #apply action
        state_2d = self.cur_state.reshape(self.origin_env_np.shape)
        if status == 0:
            state_2d[feature_idx,state_y,state_x] = 0
        else:
            state_2d[feature_idx,state_y,state_x] = 1
        self.cur_state = state_2d.reshape(-1)

        #cal reward
        reward = self.get_reward(state_2d)

        #done?
        self.step_count += 1
        done = truncated = self.step_count>=self.max_step_count
        return(
            np.array(self.cur_state,dtype=np.int32),
            reward,
            done,
            truncated,
            {}
        )


    def get_reward(self,env_arr):
        '''
        env_arr: 3D array,dim0:categoty of env,dim1:y_coord,dim2:x_coord
        '''
        return self.world.CalActionReward(env_arr)
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
    print(f"=====Iter: {i}; avg.reward={results['episode_reward_mean']}=====")