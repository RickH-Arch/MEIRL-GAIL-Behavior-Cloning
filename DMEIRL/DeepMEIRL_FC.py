import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from DMEIRL.value_iteration import value_iteration
import numpy as np
import os
from utils import utils
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


seed = 110
torch.manual_seed(seed)  # 为CPU设置随机种子
np.random.seed(seed)  # Numpy module.
if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class DeepMEIRL_FC(nn.Module):
    def __init__(self,n_input,lr = 0.001, layers = (400,300), name='deep_irl_fc') -> None:
        """initialize DeepIRl, construct function between feature and reward
        """
        super(DeepMEIRL_FC,self).__init__()
        self.n_input = n_input
        self.lr = lr
        self.name = name
        self.exploded = False

        self.net = []
        for l in layers:
            self.net.append(nn.Linear(n_input,l))
            self.net.append(nn.Tanh())
            n_input = l
        self.net.append(nn.Linear(n_input,1))
        self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)
        
        #self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=l2)#weight_decay = l2(权值衰减)
        
    
    def forward(self,features):
        out = self.net(features)
        return out
    
class DMEIRL:
    def __init__(self,world,layers = (50,30),load = "",lr = 0.001,weight_decay = 1,clip_norm = -1,log = ''):
        self.clip_norm = clip_norm
        
        self.world = world
        self.trajs = world.experts.trajs
        self.features = torch.from_numpy(world.features_arr).float().to(device)
        
        self.dynamics = torch.from_numpy(np.transpose(world.dynamics_fid,(2,1,0))).float().to(device)

        self.model = DeepMEIRL_FC(self.features.shape[1],layers = layers)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_decay)#momentum 动量,weight_decay = l2(权值衰减)

        if load != "":
            self.model.load_state_dict(torch.load(load))

        self.writer = None
        if log != "":
            self.writer = SummaryWriter(f"./run/DMEIRL_{log}")

    def train(self,n_epochs, save = True, demo = False,showInfo = False):
        self.rewards = []
        svf = self.StateVisitationFrequency()
        com_last = 100
        rewards = self.model(self.features).flatten()
        self.rewards.append(rewards.detach().cpu().numpy())
        self.exploded = False
        
        for i in range(n_epochs):
            if not demo:
                print("=============================epoch{}=============================".format(i+1))
            if i != 0:
                rewards = self.model(self.features).flatten()
                self.rewards.append(rewards.detach().cpu().numpy())

            #save rewards
            if save and not demo:
                self.SaveRewards(i)

            #show compare 
            # if i > 0 and not demo:
            #     com_last = self.CompareRewards(com_last)
            #     if showInfo:
            #         print(f"compare: {com_last}")

            #compute grad
            policy = value_iteration(0.0001,self.world,rewards.detach(),self.world.discount)
            exp_svf = self.Expected_StateVisitationFrequency(policy)
            r_grad = svf - exp_svf
            r_grad_np = r_grad.detach().cpu().numpy()
            r_grad_np = r_grad_np.__abs__()
            svf_delta = np.mean(r_grad_np)
            print("svf delta:",svf_delta)
            if self.writer:
                self.writer.add_scalar('SVF delta/Train',svf_delta,i)

            #update model
            self.optimizer.zero_grad()
            rewards.backward(-r_grad)
            if self.clip_norm != -1:
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm, norm_type=2)
                if not demo:
                    print("total_norm:",total_norm)
            self.optimizer.step()
            
            #save model
            if save and not demo:
                self.SaveModel(i)

        with torch.no_grad():
            rewards = self.model.forward(self.features).flatten()

        return rewards.detach().cpu().numpy()

    def StateVisitationFrequency(self):
        svf = torch.zeros(self.world.n_states_active,dtype=torch.float32).to(device)
        for traj in self.trajs:
            for s , *_ in traj:
                index = self.world.state_fid[s]
                svf[index] += 1
        return svf/len(self.trajs)
    
    def Expected_StateVisitationFrequency(self,policy):
        #probability of visiting the initial state
        
        #print("Expected_StateVisitationFrequency start")
        with torch.no_grad():
            prob_initial_state = torch.zeros(self.world.n_states_active,dtype=torch.float32).to(device)
            for traj in self.trajs:
                index = self.world.state_fid[traj[0][0]]
                prob_initial_state[index] += 1
            prob_initial_state = prob_initial_state/self.world.experts.trajs_count

            #Compute 𝜇
            mu = prob_initial_state.repeat(self.world.experts.traj_avg_length,1)
            x = (policy[:,:,np.newaxis]*self.dynamics).sum(1)
            for t in range(1,self.world.experts.traj_avg_length):
                mu[t,:] = torch.matmul(mu[t-1,:],x)

        return mu.sum(dim = 0)
        
    
    def SaveRewards(self,epoch):
        last_file = f"wifi_track_data/dacang/train_data/rewards_{self.model.name}_epoch{epoch}_{utils.date}.npy"
        if os.path.exists(last_file):
            os.remove(last_file)
        np.save(f"wifi_track_data/dacang/train_data/rewards_{self.model.name}_epoch{epoch+1}_{utils.date}.npy" ,self.rewards)
    
    def SaveModel(self,epoch):
        last_model = f"wifi_track_data/dacang/train_data/model_{self.model.name}_epochs{epoch}_{utils.date}.pth"
        if os.path.exists(last_model):
            os.remove(last_model)
        torch.save(self.model.state_dict(),f"wifi_track_data/dacang/train_data/model_{self.model.name}_epochs{epoch+1}_{utils.date}.pth")

    def CompareRewards(self,com_last):
        compare = nn.MSELoss()
        with torch.no_grad():
            com = compare(torch.from_numpy(self.rewards[-1]).float(),torch.from_numpy(self.rewards[-2]).float())
            com = com*100000000
            print(f"=====reward compare: {com-com_last}=====")
            if com - com_last > 1 and not self.exploded:
                np.save(f"wifi_track_data/dacang/train_data/rewards_beforeExplode_{self.model.name}_epoch{i+1}_{utils.date}.npy" ,self.rewards)
                torch.save(self.model.state_dict(),f"wifi_track_data/dacang/train_data/model_beforeExploded_{self.model.name}_epochs{i+1}_{utils.date}.pth")
                self.exploded = True
        return com



    