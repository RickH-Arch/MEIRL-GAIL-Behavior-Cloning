import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import utils
from grid_world import grid_utils,grid_plot
import math
from datetime import datetime


class GridWorld:
    def __init__(self, df_wifipos,df_path,width = 100,height = 75) -> None:
        self.empty_grid = np.zeros((height,width))
        self.count_grid = np.zeros((height,width))
        self.freq_grid = np.zeros((height,width))
        self.df_wifipos = df_wifipos
        self.df_path = df_path

        current_time = datetime.now()
        self.date = str(current_time.month)+str(current_time.day)
        
    def RecordPathCount(self,df,scale = 1):
        mac_list = df.m.unique()
        for m in tqdm(mac_list):
            df_now = utils.GetDfNow(df,m)
            x,y,z = utils.GetPathPointsWithUniformDivide(df_now,self.df_wifipos,self.df_path)
            for i in range(len(x)-1):
                point1 = (math.floor(x[i]*scale),math.floor(y[i]*scale))
                point2 = (math.floor(x[i+1]*scale),math.floor(y[i+1]*scale))
                self.count_grid= grid_utils.DrawPathOnGrid(self.count_grid,point1,point2)
        
        self.SavePathCount(f'wifi_track_data/dacang/grid_data/count_grid_{self.date}.npy')
        
    def ReadPathCount(self,path):
        self.count_grid = np.load(path)

    def SavePathCount(self,path):
        np.save(path,self.count_grid)

    def GetFreqGrid(self):
        self.freq_grid = self.count_grid/np.sum(self.count_grid)
        return self.freq_grid
    
    def GetActiveGrid(self,threshold = 0):
        self.active_grid = (self.count_grid>threshold).astype(int)
        return self.active_grid
    
    def CoordToState(self,coord):
        x,y = coord
        return int(y*self.width+x)
    
    def StateToCoord(self,state):
        x = state%self.width
        y = state//self.width
        return (x,y)
    
    def GetAllActiveStates(self):
        states = []
        for i in range(self.height):
            for j in range(self.width):
                if self.count_grid[i,j]>0:
                    states.append(self.CoordToState([j,i]))
        return states
    
    def GetAllStates(self):
        states = []
        for i in range(self.height):
            for j in range(self.width):
                states.append(self.CoordToState([j,i]))
        return states
    
    def GetStateActionPairs(self,df_now):
        x,y,z = utils.GetPathPointsWithUniformDivide(df_now,self.df_wifipos,self.df_path)

    def ShowGridWorld_Count(self):
        grid_plot.ShowGridWorld(self.count_grid)

    def ShowGridWorld_Freq(self):
        grid_plot.ShowGridWorld(self.freq_grid)

    def ShowGridWorld_Activated(self):
        grid_plot.ShowGridWorld(self.GetActiveGrid())

    
