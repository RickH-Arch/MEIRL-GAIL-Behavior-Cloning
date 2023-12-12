import numpy as np
import sys
sys.path.append('../plot/')
import myplot
from collections import namedtuple
import pandas as pd
import random
import math
import os

df_wifipos = pd.read_csv(os.getcwd()+'/../wifi_track_data/dacang/wifi_track_pos&traj/wifi_pos.csv')

def Normalize_arr(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr
    
def Normalize_df(df_in,cols = [],mul_index = 10):
    df = df_in.copy()
    if len(cols) == 0:
        cols = df.columns.tolist()
    for col in cols:
        df[col] = Normalize_arr(df[col])*mul_index
    
    return df

def _cut_Data_By_Thre(df,cut_list,cut_thre,cut_col_name,cut_mode):
    df_result = df.copy()
    for index,row in df_result.iterrows():
        if cut_mode == '>':
            if row[cut_col_name] > cut_thre :
                for col in cut_list:
                    df_result.at[index,col] = -1
        elif cut_mode == '<':
            if row[cut_col_name] < cut_thre :
                for col in cut_list:
                    df_result.at[index,col] = -1
    return df_result

def _add_data(df,col_name,list):
    min_samples_unique = df.min_samples.unique()
    df_3d = pd.DataFrame(columns=min_samples_unique)
    df_3d.insert(0,'eps',[])
    eps = df.eps.unique()
    min_samples = df.min_samples
    for i in range(len(eps)):
        df_now = (df[df.eps == eps[i]])
        add_dict = {"eps":eps[i]}
        for index,row in df_now.iterrows():
            add_dict.update({row.min_samples:row[col_name]})
        
        df_3d = df_3d._append(add_dict,ignore_index=True)

    Sur_Data = namedtuple("Sur_Data",['values','xAxes','yAxes','name'])
    data = Sur_Data(
                    values=df_3d.drop('eps',axis=1).values,
                    xAxes=df_3d.columns[1:],
                    yAxes=df_3d.eps,
                    name=col_name)
    list.append(data)
    
    # myplot.Surface3D(df_3d.drop('eps',axis=1).values,df_3d.eps,df_3d.columns[1:],
    #                 x_name = 'eps',y_name = "min_samples")


def ShowClusterResult(df,col_name_list,cut_thre = 0,cut_col_name = "",cut_mode = ''):
    if cut_thre != 0 :
        df_result = _cut_Data_By_Thre(df,col_name_list,cut_thre,cut_col_name,cut_mode)
    else:
        df_result = df.copy()
    data_list = []
    for name in col_name_list:
        _add_data(df_result,name,data_list)
    myplot.Surface3D_supPlot(data_list)

def GetWifiTrackDistance(wifi_a,wifi_b,df_pos):
    pp1 = df_pos[df_pos.wifi == wifi_a].iloc[0]
    pp2 = df_pos[df_pos.wifi == wifi_b].iloc[0]
    pos1 = [pp1.X,pp1.Y]
    pos2 = [pp2.X,pp2.Y]
    return round(_getDistance(pos1,pos2),2)

def _getDistance(track1_pos,track2_pos):
    x = track2_pos[0]-track1_pos[0]
    y = track2_pos[1]-track1_pos[1]
    return math.sqrt(x*x + y*y)




def GetRepeatTrack(df_now):
    del_list = []
    track_now = df_now.iloc[0].a
    ind_list = []
    end_mark = df_now.iloc[len(df_now)-1].mark
    for index,row in df_now.iterrows():
        
        if row.a == track_now:
            if row.mark == end_mark:
                if len(ind_list)>2:
                    del_list.extend(ind_list[0:len(ind_list)-1])
            else:
                ind_list.append(row.mark)
        else:
            track_now = row.a
            if len(ind_list)>1:
                del_list.extend(ind_list[0:len(ind_list)-1])
            ind_list = []
    return del_list

def DeleteRepeatTrack(df_now):
    del_list = set(GetRepeatTrack(df_now))
    df_now = df_now[df_now.mark.apply(lambda x : x not in del_list)]
    return df_now.reset_index().drop('index',axis=1)

def GetFirstTrack(df):
    del_list = []
    a_now = 0
    for index,row in df.iterrows():
        if row.a == a_now:
            del_list.append(row.mark)
        else:
            a_now = row.a
    return df[df.mark.apply(lambda x : False if x in del_list else True)]

def GetDfNow(df,mac):
    return df[df.m == mac].sort_values(by='t').reset_index().drop('index',axis=1)

def GetDfNowElimRepeat(df,mac):
    df_now = GetDfNow(df,mac)
    return DeleteRepeatTrack(df_now)


def PushValue(list,value,max_len):
    list.append(value)
    if(len(list)>max_len):
        list.pop(0)

def GetVirtualTrack(df_wifiPos_restored,activeSet):
    df_virtual = df_wifiPos_restored[df_wifiPos_restored.ID == "virtual"]
    
    for i,row in df_virtual.iterrows():
        set_now = set(map(int,row.parents.split(":")))
        if activeSet.issubset(set_now):
            return row.wifi
    return -1


