import numpy as np
import sys
from utils import myplot
from collections import namedtuple
import pandas as pd
import random
import math
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from datetime import datetime
current_time = datetime.now()
month = str(current_time.month)
if len(month) == 1:
    month = '0'+month
day = str(current_time.day)
if len(day) == 1:
    day = '0'+day
date = month+day



def Normalize_arr(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def Z_score(arr):
    arr = arr.reshape(-1,1) 
    scaler = StandardScaler()
    scaler.fit(arr)
    return scaler.transform(arr).flatten()
    
def Normalize_df(df_in,cols = [],mul_index = 10):
    df = df_in.copy()
    if len(cols) == 0:
        cols = df.columns.tolist()
    for col in cols:
        df[col] = Normalize_arr(df[col])*mul_index
    
    return df

def Normalize_2DArr(arr):
    f = arr.flatten()
    f = Normalize_arr(f)
    return f.reshape(arr.shape[0],arr.shape[1])

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

def GetWifiTrackDistance(wifi_a,wifi_b,df_pos,restore = False):
    pp1 = df_pos[df_pos.wifi == wifi_a].iloc[0]
    pp2 = df_pos[df_pos.wifi == wifi_b].iloc[0]
    if not restore:
        pos1 = [pp1.X,pp1.Y]
        pos2 = [pp2.X,pp2.Y]
    else:
        pos1 = [pp1.restored_x,pp1.restored_y]
        pos2 = [pp2.restored_x,pp2.restored_y]
    
    
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
        if index == 0:
            continue
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

def DeleteRepeateTrackByLocation(df_now,df_wifipos):
    '''
    delete repeate tracks according restored location
    '''
    last_loc = [0,0]
    del_marks = []
    del_marks_Now = []
    end_index = len(df_now)-1
    for index,row in df_now.iterrows():
        tracker = row.a
        row_loc = df_wifipos[df_wifipos.wifi == tracker].iloc[0]
        loc = [row_loc.restored_x,row_loc.restored_y]
        if index == 0:
            last_loc = loc
        elif index == end_index:
            if(len(del_marks_Now))>1:
                    del_marks.extend(del_marks_Now)
        else:
            if _getDistance(loc,last_loc)<1:
                del_marks_Now.append(row.mark)
            else:
                if(len(del_marks_Now))>1:
                    del_marks.extend(del_marks_Now[0:len(del_marks_Now)-1])
                del_marks_Now = []
                last_loc = loc
    del_marks = set(del_marks)
    return df_now[df_now.mark.apply(lambda x : x not in del_marks)].reset_index().drop('index',axis=1)

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
    df_virtual = df_wifiPos_restored[df_wifiPos_restored.ID.apply(lambda x : x.__contains__("virtual"))]
    
    for i,row in df_virtual.iterrows():
        set_now = set(map(int,row.parents.split(":")))
        if activeSet.issubset(set_now):
            return row.wifi
    return -1

def DeleteDriftingTrack(df_now):
    del_marks = GetDriftingTrackMarks(df_now)
    df_now = df_now[df_now.mark.apply(lambda x : x not in del_marks)].reset_index(drop = True)
    return DeleteRepeatTrack(df_now)

def GetDriftingTrackMarks(df_now):
    '''
    来回时间小于20s被认为是漂移轨迹
    '''
    m1 = -1
    m2 = -1
    t1 = 0
    t2 = 0
    del_marks = set([])
    for i,row in df_now.iterrows():
        if m1 == -1:
            m1 = row.a
        elif m2 == -1:
            m2 = row.a
            if row.a != m1:
                t1 = row.t - df_now.iloc[i-1].t
        else:
            if row.a == m1:
                t2 = row.t - df_now.iloc[i-1].t
                if (t1+t2).total_seconds()<20:
                    del_marks.add(df_now.iloc[i-1].mark)
                    m1 = -1
                    m2 = -1
                else:
                    m1 = m2
                    m2 = row.a
                    t1 = t2
            else:
                m1 = m2
                m2 = row.a
                t1 = t1 = row.t - df_now.iloc[i-1].t
    return del_marks

def GetRestoredLocation(df_wifipos,wifi):
    return df_wifipos[df_wifipos.wifi == wifi].iloc[0][['restored_x','restored_y']].tolist()

def DfToRowList(df,col_names):
    row_list = []
    for index,row in df.iterrows():
        row_list.append(row[col_names].tolist())
    return row_list

def GetKmeansClusterNumDf(df,col_names):
    X = np.array(DfToRowList(df,col_names))
    n_clusters_list = []
    calinski = []
    silhouette = []
    for n_clusters in range(2,10):
        n_clusters_list.append(n_clusters)
        kmeans = KMeans(n_clusters=n_clusters,init="k-means++",max_iter=300, random_state=0).fit(X)
        labels = kmeans.labels_
        calinski.append(calinski_harabasz_score(X, labels))
        silhouette.append(silhouette_score(X, labels))
        
    df_score = pd.DataFrame({'n_clusters':n_clusters_list,
                            'calinski':calinski,
                            'silhouette':silhouette})
    return df_score

def Kmeans(df,col_names,cluster_num):
    X = np.array(DfToRowList(df,col_names))
    kmeans = KMeans(n_clusters=cluster_num,init="k-means++",max_iter=300, random_state=0).fit(X)
    return kmeans.labels_

def _getPath(start,end,df_path):
        for i,row in df_path.iterrows():
            starts,destis = row.path.split('->')
            starts = starts.split(':')
            destis = destis.split(':')
            if str(start) in starts and str(end) in destis:
                return [x for x in row[1:] if str(x) != 'nan']
        return None

def GetPathPoints(df_now,df_wifipos,df_path):
    '''
    return all points of the incoming df_now's path
    '''
    def _append_pos(df_wifipos,tracker,time):
        z.append(time)
        x.append(df_wifipos[df_wifipos.wifi == tracker].iloc[0].restored_x)
        y.append(df_wifipos[df_wifipos.wifi == tracker].iloc[0].restored_y)

    def _append_path(df_path,start,end,start_time,end_time):
        path_points = _getPath(start,end,df_path)
        
        if len(path_points) == 0:
            return
        length = len(path_points)
        for i,point in enumerate(path_points):
            z.append(start_time + (i+1)*(end_time-start_time)/(length+1))
            xx,yy = point.split(':')
            x.append(float(xx))
            y.append(float(yy))

    x = []
    y = []
    z = []
    wifi_last = -1
    for index,row in df_now.iterrows():
        if index == 0:
            wifi_last = row.a
            continue  
        if row.a == wifi_last:
            _append_pos(df_wifipos,row.a,row.t.hour+(row.t.minute/60))
        else:
            row_last = df_now.iloc[index-1]
            time_start = row_last.t.hour+(row_last.t.minute/60)
            time_end = row.t.hour+(row.t.minute/60)
            _append_path(df_path,row_last.a,row.a,time_start,time_end)
            wifi_last = row.a
    return x,y,z

def GetPathPointsWithUniformDivide(df_now,df_wifipos,df_path):
    '''
    return all points with relevant uniform divide of the incoming df_now's path
    '''
    def _append_pos(df_wifipos,tracker,start_time,end_time):
        gap = 10/60
        inter_num = math.ceil((end_time-start_time)/gap)
        for i in range(1,inter_num):
            z.append(start_time + (i+1)*gap)
            x.append(df_wifipos[df_wifipos.wifi == tracker].iloc[0].restored_x)
            y.append(df_wifipos[df_wifipos.wifi == tracker].iloc[0].restored_y)

    def _append_path(df_path,start,end,start_time,end_time):
        path_points = _getPath(start,end,df_path)
        
        if len(path_points) == 0:
            return
        length = len(path_points)
        for i,point in enumerate(path_points):
            z.append(start_time + (i+1)*(end_time-start_time)/(length+1))
            xx,yy = point.split(':')
            x.append(float(xx))
            y.append(float(yy))

    x = []
    y = []
    z = []
    wifi_last = -1
    for index,row in df_now.iterrows():
        if index == 0:
            wifi_last = row.a
            continue  
        if row.a == wifi_last:
            row_last = df_now.iloc[index-1]
            time_start = row_last.t.hour+(row_last.t.minute/60)
            time_end = row.t.hour+(row.t.minute/60)
            _append_pos(df_wifipos,row.a,time_start,time_end)
        else:
            row_last = df_now.iloc[index-1]
            time_start = row_last.t.hour+(row_last.t.minute/60)
            time_end = row.t.hour+(row.t.minute/60)
            _append_path(df_path,row_last.a,row.a,time_start,time_end)
            wifi_last = row.a
    return x,y,z