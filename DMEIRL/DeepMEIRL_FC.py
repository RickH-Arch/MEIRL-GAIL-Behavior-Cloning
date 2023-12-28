import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from DMEIRL.value_iteration import value_iteration
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepMEIRL_FC(nn.Module):
    def __init__(self,n_input,lr = 0.001, layers = (400,300), l2=0.5, name='deep_irl_fc') -> None:
        """initialize DeepIRl, construct function between feature and reward
        """
        super(DeepMEIRL_FC,self).__init__()
        self.n_input = n_input
        self.lr = lr
        self.name = name

        self.net = []
        for l in layers:
            self.net.append(nn.Linear(n_input,l))
            self.net.append(nn.ReLU())
            n_input = l
        self.net.append(nn.Linear(n_input,1))
        self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)
        
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=l2)#momentum 动量,weight_decay = l2(权值衰减)
        #optimizer = optim.SGD(self.net.parameters,lr=self.lr,momentum = 0.9,weight_decay=l2)

        #Gradient Clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0,norm_type=2)
    
    def forward(self,features):
        out = self.net(features)
        return out
    
class DMEIRL:
    def __init__(self,world,layers = (50,30),discount = 0.9):
        self.world = world
        self.trajs = world.expert_trajs
        self.features = torch.from_numpy(world.features_arr).float().to(device)
        self.discount = discount
        self.dynamics = torch.from_numpy(world.dynamics).float().to(device)

        self.model = DeepMEIRL_FC(self.features.shape[1],layers = layers)
        self.model = self.model.to(device)

        self.optimizer = self.model.optimizer

    def train(self,n_epochs, save_rewards = True):
        self.rewards = []
        svf = self.StateVisitationFrequency()

        for epoch in tqdm(range(n_epochs)):
            rewards = self.model(self.features).flatten()
            if save_rewards:
                self.rewards.append(rewards.detach().cpu().numpy())

            policy = value_iteration(0.001,self.world,rewards,self.discount)
            exp_svf = self.Expected_StateVisitationFrequency(policy)
            r_grad = svf - exp_svf

            self.optimizer.zero_grad()
            rewards.backward(-r_grad)
            self.optimizer.step()

        with torch.no_grad():
            rewards = self.model.forward(self.features)
        return rewards

    def StateVisitationFrequency(self):
        svf = torch.zeros(self.world.n_states,dtype=torch.float32).to(device)
        for traj in self.trajs:
            for s , *_ in traj:
                svf[s] += 1
        return svf/len(self.trajs)
    
    def Expected_StateVisitationFrequency(self,policy):
        #probability of visiting the initial state
        prob_initial_state = torch.zeros(self.world.n_states,dtype=torch.float32).to(device)
        for traj in self.trajs:
            prob_initial_state[traj[0][0]] += 1
        prob_initial_state = prob_initial_state/self.trajs.shape[0]

        #Compute 𝜇
        mu = prob_initial_state.repeat(self.world.traj_avg_length,1)
        x = (policy[:,:,np.newaxis]*self.dynamics).sum(1)
        for t in range(1,self.world.traj_avg_length):
            mu[t,:] = torch.matmul(mu[t-1,:],x)

        return mu.sum(dim = 0)
    



    