import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from utils import utils,TrackCleaner
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
current_time = datetime.now()
date = str(current_time.month)+str(current_time.day)

track_data_path = "wifi_track_data/dacang/track_data/dacang_track_data_2_1217.csv"
#track_data_path = r"wifi_track_data\dacang\track_data\dacang_track_data_3_epoch6_1217.csv"
wifipos_path = 'wifi_track_data/dacang/pos_data/wifi_pos_origin.csv'
#wifipos_path = r"wifi_track_data\dacang\pos_data\processing_data\wifiposNew_needRestore_3_epoch6_1217.csv"
potential_wifipos_path = 'wifi_track_data/dacang/pos_data/potential_wifi_pos.csv'

df_poten = pd.read_csv('wifi_track_data/dacang/pos_data/potential_wifi_pos.csv')

df = pd.read_csv(track_data_path)
df.t = pd.to_datetime(df.t)
df_wifipos = pd.read_csv(wifipos_path)
df_wifipos['parents'] = ["N"]*len(df_wifipos)
df_wifipos['ID'] = ["N"]*len(df_wifipos)
epoch = 1


def TrackRestore(df,df_wifipos):
    '''
    return[0]: newTracker_count
    return[1]: df
    return[2]: df_wifiposNew
    '''
    current_time = str(datetime.now())
    print("---------------------------------------------------------------------")
    print('开始时间:',current_time)
    print('当前epoch:',epoch)
    print('当前数据量:',len(df))
    print('当前mac数量:',len(df.oriMac.unique()))
    print('当前dateMac数量:',len(df.m.unique()))
    print('当前探针数量:',len(df_wifipos))
    print("---------------------------------------------------------------------")


    mac_list = df.m.unique()

    #-----------数据评估 -切换次数 -切换时间 -切换距离 -切换速度-----------
    df_insight = TrackCleaner.InsightTrack(df,df_wifipos)
    df_insight.to_csv(os.getcwd()+f"/wifi_track_data/dacang/track_data/processing_data/insight_3_epoch{epoch}_{date}.csv",index=False)
    #df_insight = pd.read_csv(os.getcwd()+f"/wifi_track_data/dacang/track_data/processing_data/insight_3_epoch{epoch}_1217.csv")
    
    enforce_count_thre,count_thre,time_thre,distance_thre,speed_thre = TrackCleaner.AutoGetThreshold(df_insight)
    print(f"关门值:{enforce_count_thre}, 最小切换值:{count_thre}, 最大平均时间值:{time_thre}, 最大距离值:{distance_thre}, 最大速度值:{speed_thre}")
    
    #------------生成虚拟探针------------
    df_wifiposNew,normalTrack_list,enforceTrack_list = TrackCleaner.GenerateVirtualTrackerReturnTrackList(df,
                                    df_wifipos,
                                    enforce_count_thre=enforce_count_thre,
                                    count_thre = count_thre,
                                    time_thre = time_thre,
                                    dis_thre = distance_thre, 
                                    speed_thre = speed_thre,
                                    label = f'{epoch}_virtual')
    enforce_count = 0
    for l in enforceTrack_list:
        if len(l)>0:
            enforce_count+=1
    print(f"强制生成虚拟探针：{enforce_count}个")
    normal_count = 0
    for l in normalTrack_list:
        if len(l)>0:
            normal_count+=1
    print(f"普通生成虚拟探针：{normal_count}个")
    newTracker_count = len(df_wifiposNew) - len(df_wifipos)
    #如果没有新的虚拟探针了，则返回
    if newTracker_count == 0:
        return newTracker_count,'',''
    print(f"新生成虚拟探针：{newTracker_count}个")
    df_wifiposNew.to_csv(os.getcwd()+f"/wifi_track_data/dacang/pos_data/processing_data/wifiposNew_needRestore_3_epoch{epoch}_{date}.csv",index=False)
    
    
    #-----------替换跳动轨迹-----------
    df_new = pd.DataFrame(columns=df.columns)
    i = 0
    changed = False
    for mac in tqdm(mac_list,desc='替换跳动轨迹'):
        df_now = utils.GetDfNowElimRepeat(df,mac)
        if len(enforceTrack_list[i])+len(normalTrack_list[i]) > 0:
            df_result = TrackCleaner.JumpTrackRestoreWithTrackList(df_now,df_wifiposNew,enforceTrack_list[i],normalTrack_list[i])
            if df_result.equals(df_now) == False:
                changed = True
            df_now = df_result
        df_new = pd.concat([df_new,df_now],axis=0)
        i+=1

    if changed == False:
        return 0,'',''
    
    df = df_new.copy()
    df.to_csv(os.getcwd()+f"/wifi_track_data/dacang/track_data/processing_data/dacang_track_data_3_epoch{epoch}_{date}.csv",index=False)

    print(f"epoch{epoch}完成")
    print(f"完成时间{str(datetime.now())}")
    
    print("--------------------------------------------------------------------")
    return newTracker_count,df,df_wifiposNew
   

#------------------------------------------------------------------------------------------------

newTracker_count = 1
while newTracker_count>0:
    newTracker_count,df_new,df_wifiposNew = TrackRestore(df,df_wifipos)
    if newTracker_count>0:
        df = df_new
        df_wifipos = df_wifiposNew
        epoch+=1
    

#-------------------------------------------------------------------------------------------------
        
print("虚拟探针迭代已结束")
mac_list = df.m.unique()
#df_wifipos = pd.read_csv("wifi_track_data\dacang\pos_data\processing_data\wifiposNew_needRestore_3_epoch19_1216.csv")
#-----------还原虚拟探针至路径点-----------
with tqdm(total=len(df_wifipos),desc="还原虚拟探针至路径点") as pbar:
    for index,row in df_wifipos.iterrows():
        pbar.update(1)
        loc = [row['X'],row['Y']]
        dis = 1000000
        ind = -1
        for index2,row2 in df_poten.iterrows():
            loc2 = [row2['X'],row2['Y']]
            dis_now = utils._getDistance(loc,loc2)
            if dis_now < dis:
                dis = dis_now
                ind = index2
        df_wifipos.at[index,'restored_x'] = df_poten.at[ind,'X']
        df_wifipos.at[index,'restored_y'] = df_poten.at[ind,'Y']
df_wifipos.to_csv(os.getcwd()+f"/wifi_track_data/dacang/pos_data/processing_data/wifiposNew_restored_3_{date}.csv",index=False)

#-----------合并重复探针-----------
    
df_wifipos['children'] = ['N']*len(df_wifipos)
df_wifipos['activated'] = [1]*len(df_wifipos)
with tqdm(total=len(df_wifipos),desc="合并重复探针") as pbar:
    for index,row in df_wifipos.iterrows():
        pbar.update(1)
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
df_wifipos.to_csv(os.getcwd()+f"/wifi_track_data/dacang/pos_data/wifi_pos_new_3_{date}.csv",index=False)

#-----------删除重复探针-----------
#df_wifipos = pd.read_csv(r'wifi_track_data\dacang\pos_data\wifi_pos_merged_3_1218.csv')
#df = pd.read_csv(r'wifi_track_data\dacang\track_data\processing_data\dacang_track_data_3_epoch7_1218.csv')
#df.t = pd.to_datetime(df.t)
#mac_list = df.m.unique()
df_new = pd.DataFrame(columns=df.columns)
with tqdm(total=len(mac_list),desc="删除重复探针") as pbar:
    for mac in mac_list:
        pbar.update(1)
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
df_new.to_csv(os.getcwd()+f"/wifi_track_data/dacang/track_data/processing_data/dacang_track_data_3_repeateDeleted_{date}.csv",index=False)
df = df_new
print(f"当前数据量:{len(df)}")

#---------清除漂移轨迹------------
# df = pd.read_csv("wifi_track_data/dacang/track_data/dacang_track_data_3_repeateDeleted_1216.csv")
# df.t = pd.to_datetime(df.t)
df_new = pd.DataFrame(columns=df.columns)
mac_list = df.m.unique()
with tqdm(total=len(mac_list),desc="清除漂移轨迹") as pbar:
    for mac in mac_list:
        pbar.update(1)
        df_now = utils.GetDfNow(df,mac)
        if len(df_now)<5:
            continue
        df_now = utils.DeleteDriftingTrack(df_now)
        df_new = pd.concat([df_new,df_now],axis=0)
df = df_new
df.to_csv(os.getcwd()+f"/wifi_track_data/dacang/track_data/dacang_track_data_3_final_{date}.csv",index=False)

