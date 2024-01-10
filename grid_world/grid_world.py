import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import utils
from grid_world import grid_utils,grid_plot
from grid_world.experts import Experts
from datetime import datetime
import os

class GridWorld:
    '''
    class to initialize grid world,
    actions: 0:stay, 1:up, 2:down, 3:left, 4:right
    '''
    def __init__(self,
                 environments_folderPath = None,
                 features_folderPath = None,
                 states_features = None,
                expert_traj_filePath = None,
                 width = 100,height = 75,
                 trans_prob = 0.9,
                 discount = 0.9,
                 active_all = False) -> None:
        self.width = width
        self.height = height
        
        self.trans_prob = trans_prob
        self.discount = discount
        self.active_all = active_all
        
         #-------专家轨迹----------
        self.experts = Experts(expert_traj_filePath,self.width,self.height)

        #------initialize states----------
        self.count_grid = self.GetCountGrid()#每个网格被经过的次数
        self.p_grid = self.count_grid/np.sum(self.count_grid)#每个网格被经过的概率

        self.states_all = self.GetAllStates()
        self.n_states_all = len(self.states_all)

        self.states_active = self.GetAllActiveStates()
        self.n_states_active = len(self.states_active)
        self.n_actions = 5
        self.actions = [0,1,2,3,4]
        self.actions_vector = [[0,0],[0,1],[0,-1],[-1,0],[1,0]]
        self.neighbors = [0,width,-width,-1,1]

        #-------环境，特征----------
        #环境，状态-环境，环境名称列表
        if environments_folderPath:
            self.envs,self.states_envs,self.envs_list = self.ReadEnvironments(environments_folderPath)
        #特征，状态-特征，特征名称列表
        if features_folderPath:
            self.features,self.states_features,self.features_list = self.ReadFeatures(features_folderPath)
        else:
            if states_features:
                self.states_features = states_features
            else:
                raise Exception("feature_folderPath and states_features can't be None at the same time")
            
        #特征列表，转换字典
        self.features_arr,self.fid_state,self.state_fid = self.GetAvtiveFeatureArr(self.states_features)
        
        #transition probability
        self.state_adjacent_mat = self.GetStateAdjacentMat()
        self.dynamics = self.GetTransitionMat()
        #仅记录active的dynamics，系数需要经过state_fid转换
        self.dynamics_fid = self.GetTransitionMatActived()

        #get time
        current_time = datetime.now()
        self.date = str(current_time.month)+str(current_time.day)

        #helper
        self.dynamics_track = []

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
        if self.active_all:
            return self.states_all
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

#------------------------------------Init Function------------------------------------------ 
    
    def ReadEnvironments(self,folder_path):
        environments = {}
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            env_array = np.load(os.path.join(folder_path,file_name))
            environments.update({file_name.split("_")[0]:env_array})
        states_envs = {}
        for state in self.states_all:
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
        for state in self.states_all:
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
        grid_plot.ShowGridWorlds(self.envs)
    
    def ShowFeatures(self):
        grid_plot.ShowGridWorlds(self.features)

    def ShowGridWorld_Count(self):
        grid_plot.ShowGridWorld(self.count_grid)

    def ShowGridWorld_Count_log(self,title = "count_log"):
        grid_plot.ShowGridWorld(np.log(self.count_grid+1),400,400,title=title)

    def ShowGridWorld_Freq(self):
        grid_plot.ShowGridWorld(self.p_grid)

    def ShowGridWorld_Activated(self):
        grid_plot.ShowGridWorld(self.GetActiveGrid())

    def ShowRewardsResult(self,rewards,title = "Restored Rewards"):
        rewards = self.RewardsToMatrix(rewards)
        grid_plot.ShowGridWorld(rewards,400,400,title=title)

    def ShowRewardsAnimation(self,rewards,title = "Restored Rewards"):
        r = []
        for reward in rewards:
            r.append(self.RewardsToMatrix(reward))
        grid_plot.ShowGridWorld_anime(r,500,400,title=title)

    def RewardsToMatrix(self,rewards):
        rewards_matrix = np.zeros((self.height,self.width))
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
    