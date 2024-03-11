import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import utils
from grid_world import grid_utils,grid_plot
from PIL import Image
import math
from datetime import datetime
import pickle
import os


class DataParser:
    '''
    record active state, convert path to state action pairs, parse enviroment factors
    actions: 0:stay,1:up,2:down,3:left,4:right
    '''
    def __init__(self,df_wifipos = None,df_path = None,width = 100,height = 75) -> None:
        self.width = width
        self.height = height
        self.empty_grid = np.zeros((height,width))
        self.count_grid = np.zeros((height,width))
        self.freq_grid = np.zeros((height,width))
        self.df_wifipos = df_wifipos
        self.df_path = df_path
        current_time = datetime.now()
        self.date = utils.date
        self.features_dict = {}
        self.environments_dict = {}

        self.environments_arr = [] #dim0: env type, dim1: env value
        self.features_arr = [] #dim0:feature type, dim1:feature value

        #self.state_envs = {}
        #self.state_features = {}

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
                pair[0] = grid_utils.CoordToState(pair[0],self.width)
            pairs_list.append(pairs)
        pairs_dict = dict(zip(mac_list, pairs_list))
        df = pd.DataFrame({"m":mac_list,'trajs':pairs_list})
        df.to_csv(f'wifi_track_data/dacang/track_data/trajs_{self.date}_{self.width}x{self.height}.csv',index=False)
        return df
    
    def ParseEnvironmentFromFolder(self,folder_path):
        file_names = os.listdir(folder_path)
        imgs = []
        for file_name in file_names:
            imgs.append(Image.open(folder_path + "/" + file_name))
        for i in tqdm(range(len(imgs)),desc="parsing environments from folder:"):
            self.ParseEnvironmentFromImage(imgs[i],feature_name=file_names[i].split('.')[0],save_path='')
    
    def ParseEnvironments(self,image_list,feature_name_list):
        for i in range(len(image_list)):
            self.ParseEnvironmentFromImage(image_list[i],feature_name_list[i])
    
    def ParseEnvironmentFromImage(self,image,feature_name,save_path = 'wifi_track_data/dacang/grid_data'):
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
        #归一化
        #env_array = utils.Normalize_2DArr(env_array)
        #超过阈值的置为1
        env_array = np.where(env_array>10,1,0)
        #将边缘的值置为0
        for i in range(0,env_array.shape[0]):
            env_array[i,0] = 0
            env_array[i,env_array.shape[1]-1] = 0
        for i in range(0,env_array.shape[1]):
            env_array[0,i] = 0
            env_array[env_array.shape[0]-1,i] = 0
        
        self.ParseEnvironmentFrom2DArray(env_array,feature_name,save_path)
        

    def ParseEnvironmentFrom2DArray(self,env_array,feature_name='',save_path = 'wifi_track_data/dacang/grid_data'):
        '''
        env_array: 2D array, min value is 0, max value is 1
        '''
        #取得feature
        feature_array = self.__getFeatureFromEnv2DArray(env_array)

        if save_path != '' and feature_name != '':
            folder_path = os.path.join(save_path,'envs_grid',f"{self.date}_{self.width}x{self.height}")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path+f"/{feature_name}_env.npy",env_array)
            folder_path = os.path.join(save_path,'features_grid',f"{self.date}_{self.width}x{self.height}")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path+f"/{feature_name}_feature.npy",feature_array)

        if feature_name != '':
            self.environments_dict.update({feature_name:env_array})
            self.features_dict.update({feature_name:feature_array})

        self.environments_arr.append(env_array)
        self.features_arr.append(feature_array)

    def __getFeatureFromEnv2DArray(self,env_array):
        feature_array = np.zeros((env_array.shape[0],env_array.shape[1]))
        for i in range(0,feature_array.shape[0]):
            for j in range(0,feature_array.shape[1]):
                feature_array[i,j] = grid_utils.GetFeature(env_array,i,j)
        feature_array = utils.Normalize_2DArr(feature_array)
        return feature_array
    
    def GetFeaturesFromEnvs2DArray(self,env_array):
        features_arr = []
        for i in range(len(env_array)):
            features_arr.append(self.__getFeatureFromEnv2DArray(env_array[i]))
        return features_arr
    
    def Reset(self):
        self.environments_dict = {}
        self.features_dict = {}
        self.environments_arr = []
        self.features_arr = []
        

    def ShowEnvironments(self):
        grid_plot.ShowGridWorlds(self.environments_dict)
    
    def ShowFeatures(self):
        grid_plot.ShowGridWorlds(self.features_dict)

    def ShowGridWorld_Count(self):
        grid_plot.ShowGridWorld(self.count_grid)

    def ShowGridWorld_Freq(self):
        grid_plot.ShowGridWorld(self.freq_grid)

    def ShowGridWorld_Activated(self):
        grid_plot.ShowGridWorld(self.GetActiveGrid())
