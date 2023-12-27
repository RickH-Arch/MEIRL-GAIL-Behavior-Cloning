import torch
import torch.nn as nn
import torch.optim as optim


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
        
        self.optimizer = optim.Adam(self.net.parameters,lr=self.lr,momentum = 0.9,weight_decay=l2)#momentum 动量,weight_decay = l2(权值衰减)
        #optimizer = optim.SGD(self.net.parameters,lr=self.lr,momentum = 0.9,weight_decay=l2)

        #Gradient Clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0,norm_type=2)
    
    def forward(self,features):
        out = self.net(features)
        return out

    