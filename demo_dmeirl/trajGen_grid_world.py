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

class WorldObject(object):
    def __init__(self, inner_color, outer_color):
        self.inner_color = inner_color
        self.outer_color = outer_color

class GridWorld_trajGen:
    '''
    class to generate grid world trajectories according to manual set rewards
    actions: 0:stay, 1:up, 2:down, 3:left, 4:right
    '''
    def __init__(self,width,height,real_rewards_matrix,deactive_states = [],n_objects = 10,n_colors = 2,trans_prob = 0.9,discount = 0.99):
        self.width = width
        self.height = height
        self.trans_prob = trans_prob
        self.discount = discount
        self.n_objects = n_objects
        self.n_colors = n_colors
        
        self.state_now = -1

        self.states_all = self.GetAllStates()
        self.states_active = self.states_all.copy()
        for s in deactive_states:
            if s in self.states_active:
                self.states_active.remove(s)
        
        self.n_states_all= len(self.states_all)
        self.n_states_active = len(self.states_active)

        self.actions = [0,1,2,3,4]
        self.actions_vector = [[0,0],[0,1],[0,-1],[-1,0],[1,0]]
        self.n_actions = len(self.actions)
        self.neighbors = [0,width,-width,-1,1]
        self.real_rewards_matrix = real_rewards_matrix
        self.rewards_active = real_rewards_matrix.copy().reshape(self.width*self.height)[self.states_active]


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
        self.states_features = self.GetStatesFeatures()
        self.feature_arr,self.fid_state,self.state_fid = self.GetActiveFeatureArr(self.states_features)
        
        self.dynamics = self.TransitionMat()
        self.dynamics_fid = self.dynamics
        

        
        
    

    def GetAllStates(self):
        states = []
        for i in range(self.height):
            for j in range(self.width):
                states.append(self.CoordToState([j,i]))
        return states
    
    def reset(self,random = True):
        if random:
            index = np.random.randint(self.n_states_active)
            self.state = self.fid_state[index]
        else:
            self.state = 0
        return self.state
    
    def step(self, a):
        index = self.state_fid[self.state]
        probs = self.dynamics[index, a, :]
        index = np.random.choice(self.n_states_active, p=probs)
        self.state = self.fid_state[index]
        return self.state
    
    def TransitionMat(self):
        P_a = np.zeros((self.n_states_active,self.n_actions,self.n_states_active))
        for state in self.states_active:
            for a in self.actions:
                probs = self.GetTransitionStatesAndProbs(state,a)
                for next_s,prob in probs:
                    next_s = self.state_fid[next_s]
                    pre_s = self.state_fid[state]
                    P_a[pre_s,a,next_s] = prob
        return P_a
    
    def GetTransitionStatesAndProbs(self,state,action):
        if self.trans_prob == 1:
            next_s = self.LegalStateAction(state,action)
            if next_s == -1:
                #留在原地
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
        
    def OptimalPolicy(self):
        #real_rewards = torch.from_numpy(self.real_rewards_matrix.reshape(self.width*self.height)).float().to(device)
        real_rewards = torch.from_numpy(self.rewards_active).float().to(device)
        policy = value_iteration(0.0001,self,real_rewards,self.discount)
        return policy.argmax(1)
    
    def GenerateTrajectories(self,traj_count,traj_length,policy=None,save = False):
        if not policy:
            policy = self.OptimalPolicy()
        policy = policy.cpu().numpy()
        trajs = []
        for i in tqdm(range(traj_count)):
            traj = []
            state = self.reset()
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
    
    def GetStatesFeatures(self, discrete=False):
        features_arr =  np.array([self.feature_vector(i, discrete)
                         for i in range(self.n_states_all)])
        states_features = {}
        for i in range(self.n_states_all):
            states_features[i] = features_arr[i]
        
        return states_features
    
    def GetActiveFeatureArr(self,states_features):
        feature_arr = []
        fid_state = {}
        state_fid = {}
        for state,features in states_features.items():
            if state in self.states_active:
                feature_arr.append(features)
                fid_state[len(fid_state)] = state
                state_fid[state] = len(state_fid)
        return np.array(feature_arr),fid_state,state_fid
        
#------------------------------------utils method------------------------------------------
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
    
    def CoordToState(self,coord):
        x,y = coord
        return int(y*self.width+x)
    
    def StateToCoord(self,state):
        x = state%self.width
        y = state//self.width
        return (x,y)
    
#------------------------------------show method------------------------------------------
    def ShowRewards(self,title = "Rewards"):
        grid_plot.ShowGridWorld(self.real_rewards_matrix,400,400,title=title)