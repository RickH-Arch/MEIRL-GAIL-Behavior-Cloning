import gymnasium as gym
import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')
from grid_world.envGen_grid_world import GridWorld_envGen
from ray.rllib.env.env_context import EnvContext

class RegionSensor(gym.Env):
    def __init__(self,config:EnvContext,custom_config = None):
        '''
        config: width,height,envs_img_folder_path,
        target_svf_delta: dict, key:state_active, value:delta,
        model_path: path of model that used by env gen world to convert feature to reward,
        max_step_count: 
        '''

        #ray config
        if custom_config is None:
            self.width = config['width']
            self.height = config['height']
            self.envs_img_folder_path = config['envs_img_folder_path']
            self.target_svf_delta = config['target_svf_delta']
            self.model_path = config['model_path']
            self.max_step_count = config['max_step_count']
            self.experts_traj_path = config['experts_traj_path']
        #custom config
        else:
            self.width = custom_config['width']
            self.height = custom_config['height']
            self.envs_img_folder_path = custom_config['envs_img_folder_path']
            self.target_svf_delta = custom_config['target_svf_delta']
            self.model_path = custom_config['model_path']
            self.max_step_count = custom_config['max_step_count']
            self.experts_traj_path = custom_config['experts_traj_path']

        self.world = GridWorld_envGen(self.width,self.height,
                                       self.envs_img_folder_path,
                                       self.experts_traj_path,
                                       self.target_svf_delta,
                                       self.model_path)
        
        self.origin_env_np = np.array(self.world.parser.environments_arr)
        self.feature_num = self.origin_env_np.shape[0]
        self.y_num = self.origin_env_np.shape[1]
        self.x_num = self.origin_env_np.shape[2]

        self.init_state = self.origin_env_np
        self.total_state_num = len(self.init_state.reshape(-1))
        self.cur_state = self.init_state
        #self.action_space = gym.spaces.MultiDiscrete([self.origin_env_np.shape[0],self.origin_env_np.shape[1],self.origin_env_np.shape[2],2])
        self.action_space = gym.spaces.Discrete(self.total_state_num)
        self.observation_space = gym.spaces.Box(low=0,high=1,shape=(self.feature_num,self.y_num,self.x_num,),dtype=np.int32)
        self.step_count = 0
        self.reset(seed=self.width*self.height)

    def reset(self,*,seed=None,options=None):
        self.cur_state = self.init_state
        self.step_count = 0
        return np.array(self.cur_state,dtype=np.int32),{}
    
    def step(self,action):
        '''
        action: [status*feature_num*y_num*x_num + (feature_idx)*y_num*x_num + y_idx*x_num + x_idx]
        if states == 0, env = 0,
        if states == 1, env = 1,
        '''
        #parse action
        status = 0 if action <= self.total_state_num-1 else 1
        if status == 1:
            action -= self.total_state_num
        feature_idx = action // (self.y_num*self.x_num)
        idx = action - feature_idx*(self.y_num * self.x_num)
        y_idx = idx // self.x_num
        x_idx = idx - self.x_num*y_idx

        #apply action
        #state_2d = self.cur_state.reshape(self.origin_env_np.shape)
        if status == 0:
            self.cur_state[feature_idx,y_idx,x_idx] = 0
        else:
            self.cur_state[feature_idx,y_idx,x_idx] = 1

        #cal reward
        reward = self.get_reward(self.cur_state)

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
        return self.world.CalActionReward(env_arr)*10
        





