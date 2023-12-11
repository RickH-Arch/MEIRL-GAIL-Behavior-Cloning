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


def AddTrackCoupleToDf(df,wifi_a,wifi_b,switch_t,switch_speed):
    '''
    Add new couple if not exist, increace count if exist, also record couple's distance
    '''
    wifi_a = int(wifi_a)
    wifi_b = int(wifi_b)
    
    #df = pd.DataFrame({'wifi_a':[],'wifi_b':[],'count':[],'distance':[]})
    get = False
    
    for index,row in df.iterrows():
        if (row['wifi_a'] == wifi_a and row['wifi_b'] == wifi_b) or (row['wifi_a'] == wifi_b and row['wifi_b'] == wifi_a):
            get = True
            df.at[index,'count'] += 1
            df.at[index,'meanTime'] += switch_t
            if(df.at[index,'maxSpeed'] < switch_speed):
                df.at[index,'maxSpeed'] = switch_speed
            return df
    if get == False:
        dis = GetWifiTrackDistance(wifi_a,wifi_b,df_wifipos)
        df = df._append({'wifi_a':wifi_a,'wifi_b':wifi_b,'count':1,'distance':dis,'meanTime':switch_t,'maxSpeed':switch_speed},ignore_index = True)
    return df

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

def GetJumpWifiTrack(df_now,count_thre,time_thre,dis_thre,speed_thre):
    #get track switch count
    last_track = 0
    df_count = pd.DataFrame({'wifi_a':[],'wifi_b':[],'count':[]})

    for index,row in df_now.iterrows():
        if last_track == 0:
            last_track = row.a
            continue
        if last_track != row.a:
            t = df_now.iloc[index].t-df_now.iloc[index - 1].t 
            dis = GetWifiTrackDistance(df_now.iloc[index].a,df_now.iloc[index-1].a,df_wifipos)
            seconds = t.total_seconds() if t.total_seconds() > 0 else 0.5
            speed = dis/seconds
            df_count = AddTrackCoupleToDf(df_count,last_track,row.a,t,speed)
            last_track = row.a
    
    #get tracks that switch more than count_thre
    df_count = df_count[df_count['count']>count_thre]
    if len(df_count) == 0:
        return df_count

    #meanTime less than time_thre
    df_count.meanTime = df_count.meanTime.apply(lambda x :x.total_seconds())
    df_count.meanTime = df_count.meanTime/df_count['count']
    df_count = df_count[(df_count.meanTime < time_thre) | (df_count['count']>50) | (df_count.maxSpeed > speed_thre)]
    if len(df_count) == 0:
        return df_count

    #distance less than dis_thre
    df_count = df_count[(df_count.distance < dis_thre) | (df_count['count']>50) | (df_count.maxSpeed > speed_thre)]
    if len(df_count) == 0:
        return df_count
    
    #max speed > 4

    df_count = df_count[df_count.maxSpeed > 4 | (df_count['count']>50)]
    if len(df_count) == 0:
        return df_count
    
    df_count = df_count.sort_values(by='count',ascending=False)
    return df_count

def GetJumpTrackSets(track_list1,track_list2):
    '''
    get pair of track lists and return jump track sets.
    length of list1 and list2 must equal.
    a set length is with max length of 3.
    '''
    if len(track_list1) != len(track_list2):
        return
    track_sets = []
    for i in range(len(track_list1)):
        a = int(track_list1[i])
        b = int(track_list2[i])
        added = False
        for track_set in track_sets:
            # if a in track_set or b in track_set:
            #     if len(track_set) == 3:
            #         if a in track_set and b in track_set:
            #             added = True
            #         continue
            #     track_set.add(a)
            #     track_set.add(b)
            #     added = True

            #find if there are potential triangle set
            if a in track_set and len(track_set) == 2:
                c = _getTrackSetAnotherTrack(track_set,a)
                for other_set in track_sets:
                    if c in other_set and b in other_set:
                        #find new triangle
                        track_sets.remove(track_set)
                        track_sets.remove(other_set)
                        track_sets.append(set([a,b,c]))
                        added = True
                        break
        if added == False:
            track_sets.append(set([a,b]))
    return track_sets

def _getTrackSetAnotherTrack(track_set,a):
    '''
    return a *two value* track_set's another track
    '''
    if len(track_set) > 2:
        return 0
    l = list(track_set)
    return l[0] if l[1] == a else l[1]


def AddNewWifiTrack(df_wifiposNew,jumpTrack_sets):
    for track_set in jumpTrack_sets:
        #check if existed already
        info = ':'.join(map(str,track_set))
        if info in df_wifiposNew.parents.values:
            continue
        
        #add new track
        newTrack = int(round(random.random(),5)*100000)
        xx = 0
        yy = 0
        for track in track_set:
            xx += df_wifiposNew[df_wifiposNew.wifi == track].iloc[0].X
            yy += df_wifiposNew[df_wifiposNew.wifi == track].iloc[0].Y
        
        xx = int(xx/len(track_set))
        yy = int(yy/len(track_set))
        df_wifiposNew = df_wifiposNew._append({'wifi':newTrack,'X':xx,'Y':yy,'parents':info},ignore_index=True)
    return df_wifiposNew

def Show3DTrack_Origin(df_track,df_wifiPos):
    z = []
    x = []
    y = []
    for index,row in df_track.iterrows():
        z.append(row.t.hour+(row.t.minute/60))
        x.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].X)
        y.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].Y)
    myplot.Track_3D(x,y,z)

def PushValue(list,value,max_len):
    list.append(value)
    if(len(list)>max_len):
        list.pop(0)


