import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import utils
from grid_world import grid_utils,grid_plot
import math
from datetime import datetime
import pickle
import os
from datetime import datetime
current_time = datetime.now()
date = str(current_time.month)+str(current_time.day)


class GridWorld:
    '''
    class to initialize grid world,
    actions: 0:stay,1:up,2:down,3:left,4:right
    '''
    def __init__(self,
                 count_grid_filePath,
                 environments_folderPath,
                 features_folderPath,
                expert_traj_filePath,
                 width = 100,height = 75,
                 trans_prob = 0.9) -> None:
        self.width = width
        self.height = height
        
        self.trans_prob = trans_prob
        
        self.states = self.GetAllStates()
        self.n_states = len(self.states)
        self.n_actions = 5
        self.actions = [0,1,2,3,4]
        self.neighbors = [0,width,-width,-1,1]
        
        self.count_grid = np.load(count_grid_filePath)#每个网格被经过的次数
        self.p_grid = self.count_grid/np.sum(self.count_grid)#每个网格被经过的概率
        #环境，状态-环境，环境列表
        self.envs,self.states_envs,self.envs_list = self.ReadEnvironments(environments_folderPath)
        #特征，状态-特征，特征列表
        self.features,self.states_features,self.features_list = self.ReadFeatures(features_folderPath)
        self.features_arr = np.array(list(self.states_features.values()))
        #专家轨迹
        self.df_expert_trajs = self.ReadExpertTrajs(expert_traj_filePath)
        self.expert_trajs = self.df_expert_trajs['trajs'].tolist()
        self.traj_avg_length = np.mean(self.df_expert_trajs['trajs'].apply(lambda x:len(x)))

        

        #transition probability
        self.dynamics = self.GetTransitionMat()

        #get time
        current_time = datetime.now()
        self.date = str(current_time.month)+str(current_time.day)

#------------------------------------Get Method------------------------------------------
        
    def GetAllActiveStates(self):
        states = []
        for i in range(self.height):
            for j in range(self.width):
                if self.count_grid[i,j]>0:
                    states.append(self.CoordToState([j,i]))
        return states
    
    def GetAllStates(self):
        states = []
        for i in range(self.height):
            for j in range(self.width):
                states.append(self.CoordToState([j,i]))
        return states
    
    def GetFeaturesFromGivenState(self,state):
        return self.states_features[state]
    
    def GetTransitionMat(self):
        '''
        get transition dynamics of the gridworld

        return:
            P_a         N_STATESxN_STATESxN_ACTIONS transition probabilities matrix - 
                        P_a[s0, s1, a] is the transition prob of 
                        landing at state s1 when taking action 
                        a at state s0
        '''

        P_a = np.zeros((self.n_states,self.n_actions,self.n_states))
        for state in self.states:
            for a in self.actions:
                probs = self.GetTransitionStatesAndProbs(state,a)
                for next_s,prob in probs:
                    P_a[state,a,next_s] = prob
        return P_a

    def GetTransitionStatesAndProbs(self,state,action):
        if self.trans_prob == 1:
            inc = self.neighbors[action]
            next_s = state + inc
            if next_s not in self.states:
                return [(state,1)]
            else:
                return[(next_s,1)]
            
        else:
            mov_probs = np.zeros([self.n_actions])
            mov_probs += (1-self.trans_prob)/(self.n_actions-1)
            mov_probs[action] = self.trans_prob

            for a in self.actions:
                inc = self.neighbors[a]
                next_s = state + inc
                if next_s not in self.states:
                    mov_probs[-1] += mov_probs[a]
                    mov_probs[a] = 0

            res = []
            for a in self.actions:
                if mov_probs[a] != 0:
                    inc = self.neighbors[a]
                    next_s = state + inc
                    res.append((next_s,mov_probs[a]))
            return res


#------------------------------------Init Function------------------------------------------ 
    
    def ReadEnvironments(self,folder_path):
        environments = {}
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            env_array = np.load(os.path.join(folder_path,file_name))
            environments.update({file_name.split("_")[0]:env_array})
        states_envs = {}
        for state in self.states:
            states_envs.update({state:self._loadStateEnvs(state,environments)})
        environment_list = list(environments.keys())
        return environments,states_envs,environment_list

    def ReadFeatures(self,folder_path):
        features = {}
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            feature_array = np.load(os.path.join(folder_path,file_name))
            features.update({file_name.split("_")[0]:feature_array})
        states_features = {}
        for state in self.states:
            states_features.update({state:self._loadStateFeatures(state,features)})
        feature_list = list(features.keys())
        return features,states_features,feature_list

    


    def _readEnvironment(self,env_name,file_path):
        env_array = np.load(file_path)
        self.environments.update({env_name:env_array})
        return env_array
    
    def _readFeature(self,feature_name,file_path):
        feature_array = np.load(file_path)
        self.features.update({feature_name:feature_array})
        return feature_array
    
    def _loadStateEnvs(self,state,environments):
        x,y = self.StateToCoord(state)
        envs = []
        for env in environments.values():
            envs.append(env[y,x])
        return envs
    
    def _loadStateFeatures(self,state,features):
        x,y = self.StateToCoord(state)
        fs = []
        for feature in features.values():
            fs.append(feature[y,x])
        return fs
    
    def ReadExpertTrajs(self,file_path):
        df_expert_trajs = pd.read_csv(file_path)
        df_expert_trajs['trajs'] = df_expert_trajs['trajs'].apply(lambda x:eval(x))
        return df_expert_trajs
    
#------------------------------------utils method------------------------------------------
    def CoordToState(self,coord):
        x,y = coord
        return int(y*self.width+x)
    
    def StateToCoord(self,state):
        x = state%self.width
        y = state//self.width
        return (x,y)
    
    def GetActiveGrid(self,threshold = 0):
        self.active_grid = (self.count_grid>threshold).astype(int)
        return self.active_grid

#------------------------------------Plot------------------------------------------
    
    def ShowEnvironments(self):
        grid_plot.ShowGridWorlds(self.envs)
    
    def ShowFeatures(self):
        grid_plot.ShowGridWorlds(self.features)

    def ShowGridWorld_Count(self):
        grid_plot.ShowGridWorld(self.count_grid)

    def ShowGridWorld_Freq(self):
        grid_plot.ShowGridWorld(self.p_grid)

    def ShowGridWorld_Activated(self):
        grid_plot.ShowGridWorld(self.GetActiveGrid())

    
    