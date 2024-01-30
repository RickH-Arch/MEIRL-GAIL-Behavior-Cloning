import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import utils
from grid_world import grid_utils,grid_plot
from DMEIRL.value_iteration import value_iteration
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import math
np.random.seed(0)
from grid_world.grid_world import GridWorld

class WorldObject(object):
    def __init__(self, inner_color, outer_color):
        self.inner_color = inner_color
        self.outer_color = outer_color

class GridWorld_trajGen(GridWorld):
    '''
    class to generate grid world trajectories according to manual set rewards
    actions: 0:stay, 1:up, 2:down, 3:left, 4:right

    inputs:
    states_matrix: 2d array, rewards matrix, nan for deactived states
    init_states: list, initial states for trajectories
    features_folderPath: str, path to read features
    rewards_mul: list, rewards multiplier for each feature
    n_objects: int, number of objects in the grid world, default -1,used to generate features
    n_colors: int, number of colors in the grid world, default -1,used to generate features
    '''
    def __init__(self,width,height,
                 states_matrix = None,
                 init_states = [],
                 features_folderPath = None,
                 rewards_mul = [],
                 n_objects = -1,n_colors = -1,
                 trans_prob = 0.9,
                 discount = 0.9,
                 model = None):
        
        
        self.n_objects = n_objects
        self.n_colors = n_colors

        self.init_states = init_states
        self.state_now = -1
        self.states_features = None
        self.rewards_weight = rewards_mul
        self.height = height
        self.width = width
        
        #----features init----
        if n_objects != -1 and n_colors != -1:
            self.objects = {}
            for _ in range(self.n_objects):
                obj = WorldObject(np.random.randint(self.n_colors), 
                                np.random.randint(self.n_colors))
                while True:
                    x = np.random.randint(self.width)
                    y = np.random.randint(self.width)

                    if (x, y) not in self.objects:
                        break
                self.objects[x, y] = obj
            self.states_features = self.GetStatesFeaturesFromObjects()

        #get deactived states
        self.states_deactived = self.GetDeactivedStates(states_matrix)
       
        super().__init__(width=width,height=height,
                         features_folderPath=features_folderPath,
                         states_features=self.states_features,
                         trans_prob=trans_prob,
                         discount=discount,
                         manual_deact_states=self.states_deactived)
        
        if model:
            self.model = model
            self.model.eval()
            self.model.to(device)
            self.learn_rewards_matrix = self.GetRewardByModel()
        
        #----rewards init and deactivate states manually----
        if features_folderPath:
            if len(self.rewards_weight) != len(self.states_features[0]):
                raise ValueError("rewards_mul length not equal to states_features")
            self.real_rewards_matrix = self.GetRealRewards()
        else:
            self.real_rewards_matrix = states_matrix
        self.rewards_active = self.real_rewards_matrix.copy().reshape(self.width*self.height)[self.states_active]
        if model:
            self.learned_rewards_active = self.learn_rewards_matrix.copy().reshape(self.width*self.height)[self.states_active]
        
        
    
    def reset(self,random = True):
        if random:
            index = np.random.randint(self.n_states_active)
            self.state = self.fid_state[index]
        else:
            if len(self.init_states)>0:
                s = -1
                while s not in self.states_active:
                    index = np.random.randint(len(self.init_states))
                    s = self.init_states[index]
                self.state = s
            else:
                self.state = 0
        return self.state
    
    def step(self, a):
        index = self.state_fid[self.state]
        probs = self.dynamics_fid[index, a, :]
        index = np.random.choice(self.n_states_active, p=probs)
        self.state = self.fid_state[index]
        return self.state
    
    
        
    def OptimalPolicy(self,rewards_arr):
        #real_rewards = torch.from_numpy(self.real_rewards_matrix.reshape(self.width*self.height)).float().to(device)
        real_rewards = torch.from_numpy(rewards_arr).float().to(device)
        policy = value_iteration(0.001,self,real_rewards,self.discount)
        return policy.argmax(1)
    
    def GenerateTrajectories(self,traj_count,traj_length,policy=None,save = False):
        if not policy:
            policy = self.OptimalPolicy(self.rewards_active)
        policy = policy.cpu().numpy()
        trajs = []
        for i in tqdm(range(traj_count)):
            traj = []
            state = self.reset(random=False)
            for j in range(traj_length):
                index = self.state_fid[state]
                action = policy[index]
                next_state = self.step(action)
                traj.append((state,action,next_state))
                state = next_state
            trajs.append(traj)
        m = np.array(range(1,(len(trajs)+1)))
        df_trajs = pd.DataFrame({'m':m,'trajs':trajs})
        if save:
            df_trajs.to_csv(f'demo_expert_trajs_{utils.date}.csv',index=False)
        self.df_trajs_experts = df_trajs
        return df_trajs
    
    def GenerateTrajectoriesWithLearnedReward(self,traj_count,traj_length,policy=None,save = False):
        if not policy:
            policy = self.OptimalPolicy(self.learned_rewards_active)
        policy = policy.cpu().numpy()
        trajs = []
        for i in tqdm(range(traj_count)):
            traj = []
            state = self.reset(random=False)
            for j in range(traj_length):
                index = self.state_fid[state]
                action = policy[index]
                next_state = self.step(action)
                traj.append((state,action,next_state))
                state = next_state
            trajs.append(traj)
        m = np.array(range(1,(len(trajs)+1)))
        df_trajs = pd.DataFrame({'m':m,'trajs':trajs})
        if save:
            df_trajs.to_csv(f'demo_expert_trajs_{utils.date}.csv',index=False)
        self.df_trajs_learners = df_trajs
        return df_trajs
    
    def feature_vector(self, state, discrete=True):
        x_s, y_s = state%self.width, state//self.width

        nearest_inner = {}
        nearest_outer = {}

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.objects:
                    dist = math.hypot((x - x_s), (y - y_s))
                    obj = self.objects[x, y]
                    if obj.inner_color in nearest_inner:
                        if dist < nearest_inner[obj.inner_color]:
                            nearest_inner[obj.inner_color] = dist
                    else:
                        nearest_inner[obj.inner_color] = dist
                    if obj.outer_color in nearest_outer:
                        if dist < nearest_outer[obj.outer_color]:
                            nearest_outer[obj.outer_color] = dist
                    else:
                        nearest_outer[obj.outer_color] = dist

        for c in range(self.n_colors):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2*self.n_colors*self.width,))
            i = 0
            for c in range(self.n_colors):
                for d in range(1, self.width+1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
        else:
            state = np.zeros((2*self.n_colors))
            i = 0
            for c in range(self.n_colors):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state
    
    def GetStatesFeaturesFromObjects(self, discrete=False):
        features_arr =  np.array([self.feature_vector(i, discrete)
                         for i in range(self.n_states_all)])
        states_features = {}
        for i in range(self.n_states_all):
            states_features[i] = features_arr[i]
        
        return states_features
    
    def GetDeactivedStates(self,rewards_matrix):
        states_deactived = []
        for i in range(self.height):
            for j in range(self.width):
                if np.isnan(rewards_matrix[i,j]):
                    states_deactived.append(super().CoordToState((j,i)))
        return states_deactived
    
    def GetRealRewards(self):
        rewards = np.zeros((self.height,self.width))
        for j in range(self.height):
            for i in range(self.width):
                s = super().CoordToState((i,j))
                if s in self.states_active:
                    for k in range(len(self.states_features[s])):
                        rewards[j,i] += self.states_features[s][k]*self.rewards_weight[k]
                else:
                    rewards[j,i] = np.nan
        return rewards
    
    def GetRewardByModel(self):
        rewards = np.zeros((self.height,self.width))
        for j in range(self.height):
            for i in range(self.width):
                s = super().CoordToState((i,j))
                if s in self.states_active:
                    features = torch.tensor(self.states_features[s]).float().to(device)
                    rewards[j,i] = self.model.forward(features).item()
                else:
                    rewards[j,i] = np.nan
        return rewards
    
        
    
#------------------------------------show method------------------------------------------
    def ShowRewards(self,title = "Rewards"):
        rewards = self.real_rewards_matrix.copy()
        # for s in self.deactived_states:
        #     coord = self.StateToCoord(s)
        #     rewards[coord[1],coord[0]] = np.nan
        grid_plot.ShowGridWorld(rewards,400,400,title=title)

    def ShowLearnedRewards(self,title = 'Learned Rewards'):
        grid_plot.ShowGridWorld(self.learn_rewards_matrix,400,400,title = title)

    def ShowTrajs_Learner(self):
        ts = [] # t0:x1,t1:y1,t3:x2,t4:y2,t5:counts
        trajs = self.df_trajs_learners['trajs'].tolist()
        for traj in trajs:
            for i in range(len(traj)-1):
                t1 = traj[i]
                t2 = traj[i+1]
                x1,y1 = super().StateToCoord(t1[0])
                x2,y2 = super().StateToCoord(t2[0])
                x1 += 0.5
                y1 += 0.5
                x2 += 0.5
                y2 += 0.5
                for tt in ts:
                    if tt[0]==x1 and tt[1]==y1 and tt[2]==x2 and tt[3]==y2:
                        tt[4] += 1
                        break
                ts.append([x1,y1,x2,y2,1])
        grid_plot.ShowTraj(ts,self.width,self.height,title='复原轨迹')

    def ShowTrajs_Experts(self):
        ts = [] # t0:x1,t1:y1,t3:x2,t4:y2,t5:counts
        trajs = self.df_trajs_experts['trajs'].tolist()
        for traj in trajs:
            for i in range(len(traj)-1):
                t1 = traj[i]
                t2 = traj[i+1]
                x1,y1 = super().StateToCoord(t1[0])
                x2,y2 = super().StateToCoord(t2[0])
                x1 += 0.5
                y1 += 0.5
                x2 += 0.5
                y2 += 0.5
                for tt in ts:
                    if tt[0]==x1 and tt[1]==y1 and tt[2]==x2 and tt[3]==y2:
                        tt[4] += 1
                        break
                ts.append([x1,y1,x2,y2,1])
        grid_plot.ShowTraj(ts,self.width,self.height,title='专家轨迹')
                    