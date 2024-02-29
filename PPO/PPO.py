import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from utils import utils

# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10


#env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])


def find_smallest_divisor(x):
    for i in range(2, x):
        if x % i == 0:
            if i>3:
                return 1
            return i
    return None


class Actor(nn.Module):
    def __init__(self,shape_state,num_action,aPool_num,fc_layers):
        super(Actor, self).__init__()
        #cal avgPool kernel size
        poolSize = []
        
        s1 = shape_state[1]
        s2 = shape_state[2]
        for i in range(aPool_num):
            d1 = find_smallest_divisor(s1)
            d2 = find_smallest_divisor(s2)
            poolSize.append((d1,d2))
            s1 /= d1
            s2 /= d2

        #avgPool net
        avgPool_net = []
        for s in poolSize:
            avgPool_net.append(nn.AvgPool2d(kernel_size=s,stride=s))
        self.avgPool_net = nn.Sequential(*avgPool_net)

        #fc net
        n_input = int(shape_state[0]*s1*s2)
        print(f"Actor FC n_input:{n_input}, Actor FC n_output:{num_action}")
        fc_net = []
        for l in fc_layers:
            fc_net.append(nn.Linear(n_input,l))
            fc_net.append(nn.ReLU())
            n_input = l
        fc_net.append(nn.Linear(n_input,num_action))
        fc_net.append(nn.Softmax(dim=1))
        self.fc_net = nn.Sequential(*fc_net)
        #self.fc1 = nn.Linear(num_state, 100)
        #self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #action_prob = F.softmax(self.action_head(x), dim=1)
        x = self.avgPool_net(x)
        if len(x.shape)<4:
            x = x.reshape(-1).unsqueeze(0)
        else:
            x = x.reshape(x.shape[0],-1)
        action_prob = self.fc_net(x)

        return action_prob


class Critic(nn.Module):
    def __init__(self,shape_state,aPool_num,layers):
        super(Critic, self).__init__()
        poolSize = []
        
        s1 = shape_state[1]
        s2 = shape_state[2]
        for i in range(aPool_num):
            d1 = find_smallest_divisor(s1)
            d2 = find_smallest_divisor(s2)
            poolSize.append((d1,d2))
            s1 /= d1
            s2 /= d2
        
        #avgPool net
        avgPool_net = []
        for s in poolSize:
            avgPool_net.append(nn.AvgPool2d(kernel_size=s,stride=s))
        self.avgPool_net = nn.Sequential(*avgPool_net)

        #fc net
        n_input = int(shape_state[0]*s1*s2)
        print(f"Critic FC n_input:{n_input}, Critic FC n_output: 1")
        fc_net = []
        for l in layers:
            fc_net.append(nn.Linear(n_input,l))
            fc_net.append(nn.ReLU())
            n_input = l
        fc_net.append(nn.Linear(n_input,1))
        self.fc_net = nn.Sequential(*fc_net)
        #self.fc1 = nn.Linear(num_state, 100)
        #self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = self.avgPool_net(x)
        x = x.reshape(x.shape[0],-1)
        value = self.fc_net(x)
        #x = F.relu(self.fc1(x))
        #value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 10

    def __init__(self,shape_state,num_action,aPool_num,actor_layers,critic_layers):
        super(PPO, self).__init__()
        self.actor_net = Actor(shape_state,num_action,aPool_num,actor_layers)
        self.critic_net = Critic(shape_state,aPool_num,critic_layers)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('./PPO_result')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3,weight_decay=0.2)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3,weight_decay=0.2)
        # if not os.path.exists('../PPO_param'):
        #     os.makedirs('../PPO_param')

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        path = './PPO_result'
        names = os.listdir(path)
        for n in names:
            if 'epoch' in n:
                os.remove(path + '/' + n)
        torch.save(self.actor_net.state_dict(), f'./PPO_result/actor_net_{utils.date}_epoch{self.training_step}.pth')
        torch.save(self.critic_net.state_dict(),  f'./PPO_result/critic_net_{utils.date}_epoch{self.training_step}.pth')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        total_reward = np.array(reward).sum()
        self.writer.add_scalar('reward', total_reward, global_step=self.training_step)
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 5 ==0:
                    print('I_ep {} , train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1
        self.save_param()
        del self.buffer[:] # clear experience

    
# def main():
#     agent = PPO()
#     env = gym.make('CartPole-v0').unwrapped
#     num_state = env.observation_space.shape[0]
#     num_action = env.action_space.n
#     print(env.action_space)
#     torch.manual_seed(seed)
#     for i_epoch in range(1000):
#         state = env.reset()[0]
#         if render: env.render()

#         for t in count():
#             action, action_prob = agent.select_action(state)
#             next_state, reward, done, *_ = env.step(action)
#             trans = Transition(state, action, action_prob, reward, next_state)
#             if render: env.render()
#             agent.store_transition(trans)
#             state = next_state

#             if done :
#                 if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)
#                 #agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
#                 break

# if __name__ == '__main__':
#     main()
#     print("end")