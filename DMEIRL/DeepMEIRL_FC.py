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


# seed = 110
# torch.manual_seed(seed)  # 为CPU设置随机种子
# np.random.seed(seed)  # Numpy module.
# if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)


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
            self.net.append(nn.ELU())
            n_input = l
        self.net.append(nn.Linear(n_input,1))
        #self.net.append(nn.Tanh())
        self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

        #Xavier Initialize
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                #nn.init.constant_(m.bias, 0.0)
    
    def forward(self,features):
        out = self.net(features)
        return out
    
class DMEIRL:
    def __init__(self,world,layers = (50,30),load = "",lr = 0.001,weight_decay = 1,clip_norm = -1,log = '',log_dir = 'run'):
        self.clip_norm = clip_norm
        
        self.world = world
        self.trajs = world.experts.trajs
        self.features = torch.from_numpy(world.features_arr).float().to(device)
        
        #self.dynamics = torch.from_numpy(np.transpose(world.dynamics_fid,(2,1,0))).float().to(device)
        self.dynamics = torch.from_numpy(world.dynamics_fid).float().to(device)
        self.model = DeepMEIRL_FC(self.features.shape[1],layers = layers)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_decay)#momentum 动量,weight_decay = l2(权值衰减)

        if load != "":
            self.model.load_state_dict(torch.load(load))

        self.writer = None
        if log != "":
            self.writer = SummaryWriter(f"./{log_dir}/DMEIRL_{log}_lr{lr}_wd{weight_decay}_clip{clip_norm}_l{layers}")

    def train(self,n_epochs, save = True, demo = False,showInfo = False):
        self.rewards = []
        svf = self.StateVisitationFrequency()
        com_last = 100
        rewards = self.model(self.features).flatten()
        self.rewards.append(rewards.detach().cpu().numpy())
        self.exploded = False

        last_mse = 10000
        last_max = 10000
        
        for i in range(n_epochs):
            if not demo:
                print("=============================epoch{}=============================".format(i+1))
            if i != 0:
                rewards = self.model(self.features).flatten()
                self.rewards.append(rewards.detach().cpu().numpy())

            #save rewards
            if save and not demo:
                self.SyncRewards(i)

            #compute grad
            policy = value_iteration(0.005,self.world,rewards.detach(),self.world.discount,demo=demo)
            exp_svf = self.Expected_StateVisitationFrequency(policy)
            r_grad = svf - exp_svf
            r_grad_np = r_grad.detach().cpu().numpy()
            r_grad_np = r_grad_np.__abs__()
            
            svf_np = svf.detach().cpu().numpy()
            exp_svf_np = exp_svf.detach().cpu().numpy()
            mse = self.CompareSVF(svf_np,exp_svf_np)
            if not demo:
                print("svf delta:",mse)
            max = np.max(r_grad_np)
            if self.writer:
                if len(self.world.real_reward_arr)>0:
                    self.writer.add_scalar('reward loss',self.CompareWithRealReward(self.rewards[-1]),i)
                self.writer.add_scalar('SVF delta max/Train',max,i)
                self.writer.add_scalar('SVF delta min/Train',np.min(r_grad_np),i)
                self.writer.add_scalar('SVF delta mse/Train',mse,i)

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
                if last_mse>mse:
                    self.SyncModel_MinMse(i,mse)
                    last_mse = mse
                if last_max>max:
                    self.SyncModel_MinMax(i,max)
                    last_max = max


        with torch.no_grad():
            rewards = self.model.forward(self.features).flatten()

        return rewards.detach().cpu().numpy(),self.rewards

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
            d = torch.from_numpy(np.transpose(self.world.dynamics_fid,(2,1,0))).float().to(device)
            mu = prob_initial_state.repeat(self.world.experts.traj_avg_length,1)
            x = (policy[:,:,np.newaxis]*self.dynamics).sum(1)
            for t in range(1,self.world.experts.traj_avg_length):
                mu[t,:] = torch.matmul(mu[t-1,:],x)

        return mu.sum(dim = 0)
    
    def Expected_StateVisitationFrequency_2(self,policy):
        # mu[s,t] is the probability of visiting state s at time t
        policy = policy.cpu().numpy()
        mu = np.zeros([self.world.n_states_active,self.world.experts.traj_avg_length])
        for traj in self.world.experts.trajs:
            index = self.world.state_fid[traj[0][0]]
            mu[index,0] += 1
        mu[:,0] = mu[:,0]/self.world.experts.trajs_count

        for s in range(self.world.n_states_active):
            for t in range(self.world.experts.traj_avg_length-1):
                mu[s,t+1] = sum([sum([mu[pre_s,t]*self.dynamics[pre_s,a1,s]*policy[pre_s,a1] for a1 in range(self.world.n_actions)])
                                 for pre_s in range(self.world.n_states_active)])
        return mu.sum(axis=1)
    
    def SyncRewards(self,epoch):
        last_file = f"train/rewards_{self.model.name}_epoch{epoch}_{utils.date}.npy"
        if os.path.exists(last_file):
            os.remove(last_file)
        np.save(f"train/rewards_{self.model.name}_epoch{epoch+1}_{utils.date}.npy" ,self.rewards)
    
    def SyncModel_MinMse(self,epoch,mse):
        path = "train/"
        file_names = os.listdir(path)
        for n in file_names:
            if 'mse' in n:
                if os.path.exists(path+'/'+n):
                    os.remove(path + '/' + n)
        torch.save(self.model.state_dict(),f"train/model_{self.model.name}_epoch{epoch+1}_mse{mse}_{utils.date}.pth")

    def SyncModel_MinMax(self,epoch,max):
        path = "train/"
        file_names = os.listdir(path)
        for n in file_names:
            if 'max' in n:
                if os.path.exists(path+'/'+n):
                    os.remove(path + '/' + n)
        torch.save(self.model.state_dict(),f"train/model_{self.model.name}_epoch{epoch+1}_max{max}_{utils.date}.pth")

    def SaveModel(self,path):
        torch.save(self.model.state_dict(),path)

    def CompareSVF(self,svf1,svf2):
        compare = nn.MSELoss()
        with torch.no_grad():
            com = compare(torch.from_numpy(svf1).float(),torch.from_numpy(svf2).float())
        return com
    
    def CompareWithRealReward(self,reward_now):
        compare = nn.MSELoss()
        with torch.no_grad():
            com = compare(torch.from_numpy(reward_now).float(),torch.from_numpy(self.world.real_reward_arr).float())
            #print(f"=====reward compare: {com}=====")
            return com



    