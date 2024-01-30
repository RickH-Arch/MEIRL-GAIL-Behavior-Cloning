import gymnasium as gym
import numpy as np
import sys
sys.path.append('../')

class RegionSensor(gym.Env):
    def __init__(self,env_list,target_svf):
        '''
        feature_list: list of labeled features,shape[0] is the length of feature categories
        target_count_list: A 1D array that shows the target pass count of each state, used to calculate reward
        '''
        self.origin_env_np = np.array(env_list)
        self.init_state = self.origin_env_np.reshape(-1)
        self.cur_state = self.init_state
        self.action_space = gym.spaces.MultiDiscrete([self.origin_env_np.shape[0],self.origin_env_np.shape[1],2])
        self.observation_space = gym.spaces.Box(low=0.,high=1.0,shape=(self.origin_env_np.shape[0]*self.origin_env_np.shape[1],),dtype=np.float32)

    def reset(self):
        self.cur_state = self.init_state
        return self.cur_state,{}
    
    def step(self,action):
        '''
        action: [feature_idx,state,status]
        if state == 0, then the feature is set to 0
        if state == 1, then the feature is set to 1
        '''
        feature_idx = action[0]
        state = action[1]
        status = action[2]
        state_2d = self.cur_state.reshape(self.origin_env_np.shape)
        if status == 0:
            state_2d[feature_idx,state] = 0
        else:
            state_2d[feature_idx,state] = 1
        self.cur_state = state_2d.reshape(-1)
        reward = self.get_reward(self.cur_state)

    def get_reward(self,state):
        '''
        state: 1D array
        '''

        state_2d = state.reshape(self.origin_env_np.shape)
        





