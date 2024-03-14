import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import utils
from grid_world import grid_utils,grid_plot
from grid_world.data_parser import DataParser

from grid_world.experts import Experts
from datetime import datetime
import os

class GridWorld:
    '''
    class to initialize grid world,
    actions: 0:stay, 1:up, 2:down, 3:left, 4:right
    '''
    def __init__(self,
                 environments_img_folderPath = None,
                 environments_arr = None,#dim0: env type, dim1: env value
                 features_folderPath = None,
                 states_features = None,
                 features_arr = None,#dim0:feature type, dim1:feature value
                expert_traj_filePath = None,
                expert_trajs = None,
                 width = 100,height = 75,
                 trans_prob = 0.6,
                 discount = 0.98,
                 active_all = False,
                 manual_deact_states = [],
                 real_reward_mat = [],
                 traj_length_bias = 0) -> None:
        self.width = width
        self.height = height
        
        self.trans_prob = trans_prob
        self.discount = discount
        self.active_all = active_all
        self.manual_deact_states = manual_deact_states
        self.traj_len_bias = traj_length_bias

        
        
        #-------专家轨迹----------
        self.experts = None
        try:
            if expert_traj_filePath:
                self.experts = Experts(self.width,self.height,trajs_file_path=expert_traj_filePath,bias=traj_length_bias)
            elif any(expert_trajs):
                self.experts = Experts(self.width,self.height,df_trajs=expert_trajs,bias = traj_length_bias)
        except:
            pass
             
        #print("experts didn't initialize")

        self.count_grid = np.zeros((self.height,self.width))
        if self.experts:
            self.count_grid = self.GetCountGrid()#每个网格被经过的次数
            #self.state_adjacent_mat = self.GetStateAdjacentMat()
        #------initialize states----------
        
        self.states_all = self.GetAllStates()
        self.n_states_all = len(self.states_all)

        self.states_active = self.GetAllActiveStates()
        self.n_states_active = len(self.states_active)
        self.n_actions = 5
        self.actions = [0,1,2,3,4]
        self.actions_vector = [[0,0],[0,1],[0,-1],[-1,0],[1,0]]
        self.neighbors = [0,width,-width,-1,1]

        #-------环境，特征----------
        self.parser = DataParser(width=self.width,height=self.height)
        #envs: 3d array, dim0: env type, [dim1,dim2]: env value
        #states_envs: dict, key: state, value: env values
        #envs_list: list of env names
        if environments_img_folderPath:
            self.parser.ParseEnvironmentFromFolder(environments_img_folderPath)
            #self.envs,self.states_envs,self.envs_list = self.ReadEnvironmentsFromFolder(environments_folderPath)
            self.envs_dict = self.parser.environments_dict
            self.envs_list = list(self.parser.environments_dict.keys())
            self.states_envs = self.GetStatesValueFromDict(self.envs_dict)
            self.features_dict = self.parser.features_dict
            self.features_list = list(self.parser.features_dict.keys())
            self.states_features = self.GetStatesValueFromDict(self.features_dict)
        else:
            if environments_arr:
                self.envs = environments_arr
                self.states_envs = self.GetStatesValueFromArr(self.envs)
                self.envs_list = None
            #特征，状态-特征，特征名称列表
            if features_folderPath:
                self.features_dict,self.states_features,self.features_list = self.ReadFeaturesFromFolder(features_folderPath)
            elif states_features:
                self.states_features = states_features
                self.features_dict = self.SplitFeatures(self.states_features)
            # elif features_arr:
            #     self.features = features_arr
            #     self.states_features = self.GetStatesValueFromArr(self.features)
            #     self.features_list = None
            
            
        #特征列表，转换字典
        if len(self.states_features)>0:
            self.features_arr,self.fid_state,self.state_fid = self.GetAvtiveFeatureArr(self.states_features)
        #transition probability
        self.dynamics = self.GetTransitionMat()
        #仅记录active的dynamics，系数需要经过state_fid转换
        self.dynamics_fid = self.GetTransitionMatActived()

        #helper
        self.dynamics_track = []

        #real reward matrix, used for evaluate trained reward
        self.real_reward_arr = self.GetRealRewardArr(real_reward_mat)
    

#------------------------------------Get Method------------------------------------------
    def GetCountGrid(self):
        count_grid = np.zeros((self.height,self.width))
        trajs = self.experts.trajs_all
        for traj in trajs:
            for t in traj:
                s = t[0]
                x,y = self.StateToCoord(s)
                count_grid[y,x] += 1
        return count_grid

    def GetAllActiveStates(self):
        states = []
        if self.active_all:
            return self.states_all.copy()
        
        #如果没有专家轨迹，那么所有不在manual_deact_states中的状态都是active
        if not self.experts:
            for s in self.states_all:
                if s not in self.manual_deact_states:
                    states.append(s)
            return states
        #如果有专家轨迹，那么所有在专家轨迹中出现过的状态都是active，且忽略manual_deact_states
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
                        P_a[s0,a,s1] is the transition prob of 
                        landing at state s1 when taking action 
                        a at state s0
        '''

        P_a = np.zeros((self.n_states_all,self.n_actions,self.n_states_all))
        for state in self.states_active:
            for a in self.actions:
                probs = self.GetTransitionStatesAndProbs(state,a)
                for next_s,prob in probs:
                    P_a[state,a,next_s] = prob
        return P_a
    
    def GetTransitionMatActived(self):
        P_a = np.zeros((self.n_states_active,self.n_actions,self.n_states_active))
        as_set = set(self.states_active)
        for s_0 in range(self.dynamics.shape[0]):
            for a in range(self.dynamics.shape[1]):
                for s_1 in range(self.dynamics.shape[2]):
                    if s_0 in as_set and s_1 in as_set:
                        P_a[self.state_fid[s_0],a,self.state_fid[s_1]] = self.dynamics[s_0,a,s_1]
        return P_a

    def GetStateAdjacentMat(self):
        '''
        get adjacent matrix of the gridworld

        return:
            adjacent_mat         N_STATESxN_STATES adjacent matrix - 
                        adjacent_mat[s0, s1] is 1 if s1 is adjacent to s0
        '''
        adjacent_mat = np.zeros((self.n_states_all,self.n_states_all))
        for s in range(self.n_states_all):
            adjacent_mat[s,s] = 1
        for traj in self.experts.trajs_all:
            for i in range(len(traj)-1):
                adjacent_mat[traj[i],traj[i+1]] = 1
                adjacent_mat[traj[i+1],traj[i]] = 1
        return adjacent_mat

    def GetTransitionStatesAndProbs(self,state,action):
        if self.trans_prob == 1:
            next_s = self.LegalStateAction(state,action)
            #如果不通或者出界，返回原地
            if next_s == -1:
                return [(state,1)]
            else:
                return[(next_s,1)]
            
        else:
            mov_probs = np.zeros([self.n_actions])
            mov_probs += (1-self.trans_prob)/(self.n_actions-1)
            mov_probs[action] = self.trans_prob

            for a in self.actions:
                next_s = self.LegalStateAction(state,a)
                if next_s == -1:
                    mov_probs[0] += mov_probs[a]
                    mov_probs[a] = 0

            res = []
            for a in self.actions:
                if mov_probs[a] != 0:
                    inc = self.neighbors[a]
                    next_s = state + inc
                    res.append((next_s,mov_probs[a]))
            return res

    def GetAvtiveFeatureArr(self,states_features):
        feature_arr = []
        fid_state = {}
        state_fid = {}
        for state,features in states_features.items():
            if state in self.states_active:
                feature_arr.append(features)
                fid_state[len(fid_state)] = state
                state_fid[state] = len(state_fid)
        return np.array(feature_arr),fid_state,state_fid
    
    #convert 2d array real reward to 1d array
    def GetRealRewardArr(self,real_reward_mat):
        if len(real_reward_mat) == 0:
            return []
        reward_arr = []
        for state in self.states_active:
            index = self.StateToCoord(state)
            reward_arr.append(real_reward_mat[index[1],index[0]])
        reward_arr = np.array(reward_arr)
        reward_arr = utils.Normalize_arr(reward_arr)
        return reward_arr

#------------------------------------Init Function------------------------------------------ 
    def SplitFeatures(self,states_features):
        features = {}
        for state,fs in states_features.items():
            for i,f in enumerate(fs):
                if i not in features:
                    features[i] = np.zeros((self.height,self.width))
                coord = self.StateToCoord(state)
                features[i][coord[1],coord[0]] = f
        return features
    
    def ShowReward(self,reward_arr):
        reward_grid = np.zeros((self.height,self.width))
        for i in range(len(reward_arr)):
            state = self.fid_state[i]
            coord = self.StateToCoord(state)
            reward_grid[coord[1],coord[0]] = reward_arr[i]
        for i in range(self.height):
            for j in range(self.width):
                if self.count_grid[i,j] == 0:
                    reward_grid[i,j] = np.nan
        grid_plot.ShowGridWorld(reward_grid,500,400,title="Restored Rewards")


    def ReadEnvironmentsFromFolder(self,folder_path):
        environments = {}
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            env_array = np.load(os.path.join(folder_path,file_name))
            environments.update({file_name.split("_")[0]:env_array})

        states_envs = self.GetStatesValueFromDict(list(environments.values()))
        
        return environments,states_envs,environments.keys()

    def ReadFeaturesFromFolder(self,folder_path):
        features = {}
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            feature_array = np.load(os.path.join(folder_path,file_name))
            features.update({file_name.split("_")[0]:feature_array})
        
        states_features= self.GetStatesValueFromDict(list(features.values()))

        return features,states_features,features.keys()
    
    def GetStatesValueFromDict(self,values_dict):
        states_value = {}
        for state in self.states_all:
            states_value.update({state:self._loadStateValue(state,values_dict)})
        
        return states_value
    
    def GetStatesValueFromArr(self,values_arr):
        states_value = {}
        for s in self.states_all:
            states_value.update({s:self._loadStateValue(s,values_arr)})
        return states_value


    def _readEnvironment(self,env_name,file_path):
        env_array = np.load(file_path)
        self.environments.update({env_name:env_array})
        return env_array
    
    def _readFeature(self,feature_name,file_path):
        feature_array = np.load(file_path)
        self.features_dict.update({feature_name:feature_array})
        return feature_array
    
    def _loadStateValue(self,state,values):
        x,y = self.StateToCoord(state)
        vs = []
        if type(values) == dict:
            values = values.values()
            for value in values:
                vs.append(value[y,x])
        else:
            for value in values:
                vs.append(value[y,x])
        return vs
    
    def ReadExpertTrajs(self,file_path):
        df_expert_trajs = pd.read_csv(file_path)
        df_expert_trajs['trajs'] = df_expert_trajs['trajs'].apply(lambda x:eval(x))
        return df_expert_trajs
    
    def SetTransitionProb(self,prob):
        self.trans_prob = prob
        #transition probability
        self.dynamics = self.GetTransitionMat()
        #仅记录active的dynamics，系数需要经过state_fid转换
        self.dynamics_fid = self.GetTransitionMatActived()
    
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
    
    def LegalStateAction(self,state,action):
        inc = self.neighbors[action]
        dir = self.actions_vector[action]
        coord = self.StateToCoord(state)
        next_s = state + inc
        next_coord = (coord[0] + dir[0], coord[1] + dir[1])
        if next_coord[0]<0 or next_coord[0]>self.width-1 or next_coord[1] < 0 or next_coord[1] > self.height-1:
            return -1
        if next_s not in self.states_active:
            return -1
        return next_s

#------------------------------------Plot------------------------------------------
    
    def ShowEnvironments(self):
        grid_plot.ShowGridWorlds(self.envs_dict)
    
    def ShowFeatures(self):
        grid_plot.ShowGridWorlds(self.features_dict)

    def ShowGridWorld_Count(self):
        grid_plot.ShowGridWorld(self.count_grid)

    def ShowGridWorld_Count_log(self,title = "count_log"):
        grid_plot.ShowGridWorld(np.log(self.count_grid+1),500,400,title=title)

    def ShowGridWorld_Freq(self):
        grid_plot.ShowGridWorld(self.p_grid)

    def ShowGridWorld_Activated(self):
        grid_plot.ShowGridWorld(self.GetActiveGrid(),width=600,title='Actived Grid World')

    def ShowRewardsResult(self,rewards,title = "Restored Rewards"):
        rewards = self.RewardsToMatrix(rewards)
        grid_plot.ShowGridWorld(rewards,400,400,title=title)

    def ShowRewardsAnimation(self,rewards,title = "Restored Rewards"):
        r = []
        for reward in rewards:
            r.append(self.RewardsToMatrix(reward))
        grid_plot.ShowGridWorld_anime(r,480,400,title=title)

    def ShowSVF(self,svf,title):
        SVF_total = np.zeros((self.height,self.width))
        for s in range(len(svf)):
            s_now = self.fid_state[s]
            x,y = grid_utils.StateToCoord(s_now,self.width)
            SVF_total[y,x] = svf[s]
        grid_plot.ShowGridWorld(SVF_total,title=title)

    def ShowGridValue(self,value,title):
        value_total = np.zeros((self.height,self.width))
        for s in range(len(value)):
            s_now = self.fid_state[s]
            x,y = grid_utils.StateToCoord(s_now,self.width)
            value_total[y,x] = value[s]
        grid_plot.ShowGridWorld(value_total,title=title)

    def ShowGrid3DBarChart(self,value,title):
        coords = []
        for i in range(len(value)):
            s = self.fid_state[i]
            x,y = grid_utils.StateToCoord(s,self.width)
            coords.append([x*10,y*10])
        grid_plot.BarChart_3D(coords,value)

    def RewardsToMatrix(self,rewards):
        rewards_matrix = np.zeros((self.height,self.width))
        for i in range(self.height):
            for j in range(self.width):
                if rewards_matrix[i,j] == 0:
                    rewards_matrix[i,j] = np.nan
        for i in range(len(rewards)):
            state = self.fid_state[i]
            coord = self.StateToCoord(state)
            rewards_matrix[coord[1],coord[0]] = rewards[i]
        
        return rewards_matrix


#--------------------------------helper mathod------------------------------------------
    
    def ShowAllActiveStatesPosition(self):
        states_position = np.zeros((self.height,self.width))
        for state in self.states_active:
            x,y = self.StateToCoord(state)
            states_position[y,x] = 1
        grid_plot.ShowGridWorld(states_position)

    def ShowDynamics(self,dir):
        if len(self.dynamics_track) == 0:
            dynamic_track = [[],[],[],[],[]]
            for i in range(self.dynamics.shape[0]):
                x_start,y_start = self.StateToCoord(i)
                for j in range(self.dynamics.shape[1]):
                    for k in range(self.dynamics.shape[2]):
                        if self.dynamics[i,j,k] != 0:
                            x_end,y_end = self.StateToCoord(k)
                            
                            dynamic_track[j].append([x_start,y_start,x_end,y_end,self.dynamics[i,j,k]])
            self.dynamics_track = dynamic_track

        grid_plot.ShowDynamics(self.dynamics_track,dir,self.width,self.height,self.GetActiveGrid())
    