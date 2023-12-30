import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import utils
from grid_world import grid_utils,grid_plot
import math
from datetime import datetime
import pickle
import os
from datetime import datetime
current_time = datetime.now()
date = str(current_time.month)+str(current_time.day)

class DataParser:
    '''
    record active state, convert path to state action pairs, parse enviroment factors
    actions: 0:stay,1:up,2:down,3:left,4:right
    '''
    def __init__(self,df_wifipos,df_path,width = 100,height = 75) -> None:
        self.width = width
        self.height = height
        self.empty_grid = np.zeros((height,width))
        self.count_grid = np.zeros((height,width))
        self.freq_grid = np.zeros((height,width))
        self.df_wifipos = df_wifipos
        self.df_path = df_path
        current_time = datetime.now()
        self.date = str(current_time.month)+str(current_time.day)
        self.features = {}
        self.environments = {}

        # self.states = self.GetAllStates()
        # self.n_states = len(self.states)

        # self.n_actions = 5

        self.state_envs = {}
        self.state_features = {}

    def RecordPathCount(self,df,scale = 1):
        mac_list = df.m.unique()
        for m in tqdm(mac_list):
            df_now = utils.GetDfNow(df,m)
            x,y,z = utils.GetPathPointsWithUniformDivide(df_now,self.df_wifipos,self.df_path)
            for i in range(len(x)-1):
                point1 = (math.floor(x[i]*scale),math.floor(y[i]*scale))
                point2 = (math.floor(x[i+1]*scale),math.floor(y[i+1]*scale))
                self.count_grid= grid_utils.DrawPathOnGrid(self.count_grid,point1,point2)
        
        np.save(f'wifi_track_data/dacang/grid_data/count_grid_{self.date}.npy',self.count_grid)

    def PathToStateActionPairs(self,df,scale = 1):
        mac_list = df.m.unique()
        state_list = []
        for m in tqdm(mac_list):
            state = []
            df_now = utils.GetDfNow(df,m)
            x,y,z = utils.GetPathPointsWithUniformDivide(df_now,self.df_wifipos,self.df_path)
            for i in range(len(x)-1):
                point1 = (math.floor(x[i]*scale),math.floor(y[i]*scale))
                point2 = (math.floor(x[i+1]*scale),math.floor(y[i+1]*scale))
                state.extend(grid_utils.GetPathCorList(self.count_grid,point1,point2))
            state_list.append(state)
        print("Converting to state action pairs...")
        pairs_list = []
        for i in range(len(state_list)):
            states = state_list[i]
            pairs = grid_utils.StatesToStateActionPairs(states)
            for pair in pairs:
                pair[0] = self.CoordToState(pair[0])
            pairs_list.append(pairs)
        pairs_dict = dict(zip(mac_list, pairs_list))
        df = pd.DataFrame({"m":mac_list,'trajs':pairs_list})
        df.to_csv(f'wifi_track_data/dacang/track_data/trajs_{self.date}.csv',index=False)
        return df
    
    def ParseEnvironments(self,image_list,feature_name_list):
        for i in range(len(image_list)):
            self.ParseEnvironment(image_list[i],feature_name_list[i])
    
    def ParseEnvironment(self,image,feature_name):
        '''
        args[0]:the labled rgb Image
        args[1]:name of the parsing environment 
        '''
        image_array = np.array(image)
        image_array = np.invert(image_array)#反相
        image_array = np.flipud(image_array)#上下翻转

        #对image第三维进行求和
        env_array = np.zeros((image_array.shape[0],image_array.shape[1]))
        for i in range(0,image_array.shape[0]):
            for j in range(0,image_array.shape[1]):
                env_array[i,j] = np.sum(image_array[i,j,:])

        folder_path = os.path.join('wifi_track_data/dacang/grid_data/envs_grid',date)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path+f"/{feature_name}_env.npy",env_array)
        self.environments.update({feature_name:env_array})
        
        #取得feature
        feature_array = np.zeros((image_array.shape[0],image_array.shape[1]))
        for i in range(0,feature_array.shape[0]):
            for j in range(0,feature_array.shape[1]):
                feature_array[i,j] = grid_utils.GetFeature(env_array,i,j)

        feature_array = utils.Normalize_2DArr(feature_array)

        folder_path = os.path.join('wifi_track_data/dacang/grid_data/features_grid',date)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path+f"/{feature_name}_feature.npy",feature_array)
        self.features.update({feature_name:feature_array})

    def ShowEnvironments(self):
        grid_plot.ShowGridWorlds(self.environments)
    
    def ShowFeatures(self):
        grid_plot.ShowGridWorlds(self.features)

    def ShowGridWorld_Count(self):
        grid_plot.ShowGridWorld(self.count_grid)

    def ShowGridWorld_Freq(self):
        grid_plot.ShowGridWorld(self.freq_grid)

    def ShowGridWorld_Activated(self):
        grid_plot.ShowGridWorld(self.GetActiveGrid())
