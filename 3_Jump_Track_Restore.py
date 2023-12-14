import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from utils import utils,TrackCleaner
from utils.BoxFeatures import BoxFeature

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
current_date = datetime.now()
date = str(current_date.month)+str(current_date.day)

track_data_path = "wifi_track_data/dacang/track_data/dacang_track_data_2_1214.csv"
wifipos_path = 'wifi_track_data/dacang/pos_data/wifi_pos.csv'
potential_wifipos_path = 'wifi_track_data/dacang/pos_data/potential_wifi_pos.csv'

df_poten = pd.read_csv('wifi_track_data/dacang/pos_data/potential_wifi_pos.csv')

df = pd.read_csv(track_data_path)
df.t = pd.to_datetime(df.t)
df_wifipos = pd.read_csv(wifipos_path)
epoch = 1


def TrackRestore(df,df_wifipos,count_thre=0,time_thre=0,distance_thre=0,speed_thre=0):
    '''
    return[0]: newTracker_count
    return[1]: df
    return[2]: df_wifiposNew
    '''

    print("---------------------------------------------------------------------")
    print('----当前epoch-----:',epoch)
    print('----当前数据量-----:',len(df))
    print('----当前mac数量-----:',len(df.oriMac.unique()))
    print('-----当前dateMac数量-----:',len(df.m.unique()))
    print('-----当前探针数量-----:',len(df_wifipos))
    print("---------------------------------------------------------------------")


    mac_list = df.m.unique()

    #-----------数据评估 -切换次数 -切换时间 -切换距离 -切换速度-----------
    df_insight = TrackCleaner.InsightTrack(df,df_wifipos)
    df_insight.to_csv(os.getcwd()+f"/dacang/track_data/processing_data/insight_3_epoch{epoch}_{date}.csv",index=False)

    #获取切换次数临界值 - Q3
    if count_thre == 0:
        count_thre = BoxFeature(df_insight['count'])[4]

    #获取平均切换时间 - Q3+50
    if time_thre == 0:
        time_thre = BoxFeature(df_insight['meanTime'])[4]+50

    #获取满足次数下的平均切换距离 - count>count_thre -> Q3
    if distance_thre == 0:
        distance_thre = BoxFeature(df_insight[df_insight['count']>count_thre]['distance'])[4]

    #获取满足次数下的最大切换速度 - count>count_thre -> Q1
    if speed_thre == 0:
        speed_thre = BoxFeature(df_insight[df_insight['count']>count_thre]['maxSpeed'])[2]

    
    #------------生成虚拟探针------------
    df_wifiposNew = TrackCleaner.GenerateVirtualTracker(df,
                                    df_wifipos,
                                    count_thre = count_thre,
                                    time_thre = time_thre,
                                    dis_thre = distance_thre, 
                                    speed_thre = speed_thre,
                                    label = f'{epoch}_virtual')
    newTracker_count = len(df_wifiposNew) - len(df_wifipos)
    #如果没有新的虚拟探针了，则返回
    if newTracker_count == 0:
        return newTracker_count,'',''
    print(f"新生成虚拟探针：{newTracker_count}个")
    df_wifiposNew.to_csv(os.getcwd()+f"/dacang/pos_data/processing_data/wifiposNew_needRestore_3_epoch{epoch}_{date}.csv",index=False)
    
    #-----------还原虚拟探针至路径点-----------
    for index,row in df_wifiposNew.iterrows():
        loc = [row['X'],row['Y']]
        dis = 1000000
        ind = -1
        for index2,row2 in df_poten.iterrows():
            loc2 = [row2['X'],row2['Y']]
            dis_now = utils._getDistance(loc,loc2)
            if dis_now < dis:
                dis = dis_now
                ind = index2
        df_wifiposNew.at[index,'restored_x'] = df_poten.at[ind,'X']
        df_wifiposNew.at[index,'restored_y'] = df_poten.at[ind,'Y']
    
    df_wifiposNew.to_csv(os.getcwd()+f"/dacang/pos_data/processing_data/wifiposNew_restored_3_epoch{epoch}_{date}.csv",index=False)

    #-----------替换跳动轨迹-----------
    df_new = pd.DataFrame(columns=df.columns)
    for mac in tqdm(mac_list,desc='替换跳动轨迹'):
        df_now = utils.GetDfNowElimRepeat(df,mac)
        df_now = TrackCleaner.JumpTrackRestore(df_now,
                                        df_wifiposNew,
                                        count_thre = count_thre,
                                        time_thre = time_thre,
                                        dis_thre = distance_thre, 
                                        speed_thre = speed_thre)
        df_new = pd.concat([df_new,df_now],axis=0)
    df = df_new.copy()
    #df_new.to_csv(os.getcwd()+f"/dacang/track_data/dacang_track_data_3_epoch{epoch}_{date}.csv",index=False)


    #-----------合并重复探针-----------
    print("正在合并重复探针...")
    df_wifipos = df_wifiposNew.copy()
    df_wifipos['children'] = ['N']*len(df_wifipos)
    df_wifipos['activated'] = [1]*len(df_wifipos)
    for index,row in df_wifipos.iterrows():
        if row.activated == 0:
            continue
        for i in range(index+1,len(df_wifipos)):
            row_now = df_wifipos.iloc[i]
            if row_now.activated == 0:
                continue
            if utils.GetWifiTrackDistance(row.wifi,row_now.wifi,df_wifipos,True) < 1:
                df_wifipos.at[i,'activated'] = 0
                if df_wifipos.at[index,'children'] == "N":
                    df_wifipos.at[index,'children'] = ""
                df_wifipos.at[index,'children'] = df_wifipos.at[index,'children'] + (str(row_now.wifi)+':')
    df_wifipos = df_wifipos[df_wifipos.activated == 1].reset_index().drop('index',axis=1)
    df_wifipos.to_csv(os.getcwd()+f"/dacang/pos_data/processing_data/wifi_pos_delRepeate_3_epoch{epoch}_{date}.csv",index=False)

    #-----------删除重复探针-----------
    print("正在删除重复探针...")
    df_new = pd.DataFrame(columns=df.columns)
    for mac in mac_list:
        df_now = utils.GetDfNow(df,mac)
        for index,row in df_now.iterrows():
            for index2,row2 in df_wifipos.iterrows():
                children = row2.children.split(':')
                if str(row.a) in children:
                    df_now.at[index,'a'] = row2.wifi
                    break
        if len(df_now)>2:
            df_now = utils.DeleteRepeatTrack(df_now)
        df_new = pd.concat([df_new,df_now],axis=0)
    df_new.to_csv(os.getcwd()+f"/dacang/track_data/dacang_track_data_3_epoch{epoch}_{date}.csv",index=False)


    return newTracker_count,df_new,df_wifipos

newTracker_count = 1
while newTracker_count>0:
    newTracker_count,df_new,df_wifiposNew = TrackRestore(df,df_wifipos)
    if newTracker_count>0:
        df = df_new
        df_wifipos = df_wifiposNew
        epoch+=1

#---------清除漂移轨迹------------

df_new = pd.DataFrame(columns=df.columns)
mac_list = df.m.unique()
for mac in mac_list:
    df_now = utils.GetDfNow(mac)
    df_now = utils.DeleteDriftingTrack(df_now)
    df_new = pd.concat([df_new,df_now],axis=0)
df = df_new

#---------清除一直在同一地方的mac----------

list_count = df.groupby(['m']).a.value_counts()
dd = list_count.to_frame().rename(columns={'a':'A'}).reset_index()
mac_Once = dd[~dd.duplicated('m', keep=False)].m.reset_index().drop('index',axis=1)
df = df.loc[~df.m.isin(mac_Once.m)]
df.to_csv(os.getcwd()+f"/dacang/track_data/dacang_track_data_3_final_{date}.csv",index=False)