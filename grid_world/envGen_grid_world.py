import pandas as pd
import numpy as np
from grid_world.grid_world import GridWorld
from grid_world.data_parser import DataParser
from grid_world import grid_utils,grid_plot
from utils import utils
from DMEIRL.value_iteration import value_iteration
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import Image
import os

class GridWorld_envGen(GridWorld):
    def __init__(self,width,height,
                 envs_folder_path,
                 experts_traj_filePath,
                 target_svf_delta:dict,
                 model,
                 trans_prob = 0.6,
                 discount = 0.98,
                 ):
        self.width = width
        self.height = height
        self.parser = DataParser(width=self.width,height=self.height)
        self.__parseOriginEnvs(envs_folder_path)


        super().__init__(width=width,height=height,
                         expert_traj_filePath=experts_traj_filePath,
                         trans_prob=trans_prob,
                         discount=discount)
        if model:
            self.model = model
            self.model.eval().to(device)

        self.prob_initial_state = self.__getInitialStatesProb()

    def __parseOriginEnvs(self,folder_path):
        file_names = os.listdir(folder_path)
        imgs = []
        for file_name in file_names:
            imgs.append(Image.open(folder_path + "/" + file_name))
        for i in range(len(imgs)):
            self.parser.ParseEnvironmentFromImage(imgs[i],file_names[i].split('.')[0],save_path='')


        
            
    def CalActionReward(self,envs):
        pass

    def __getFeaturesFromEnvs(self,envs):
        pass

    def __getWalkingRewards(self):
        pass

    def __getSVF(self):
        pass

    def __calSVFLoss(self,exp_svf):
        pass

    def __getInitialStatesProb(self):
        prob_initial_state = torch.zeros(self.n_states_active,dtype=torch.float32).to(device)
        for traj in self.experts.trajs:
            index = self.state_fid[traj[0][0]]
            prob_initial_state[index] += 1
        prob_initial_state = prob_initial_state/self.experts.trajs_count
        return prob_initial_state