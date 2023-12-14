import numpy as np
import pandas as pd
import random
import math
import os
from utils import utils
from tqdm import tqdm


def JumpTrackRestore(df_now,
                     df_wifipos,
                     count_thre,
                     time_thre,
                     dis_thre, 
                     speed_thre):

    df_count = GetJumpWifiTrackCouple(df_now,
                                      df_wifipos,
                                      count_thre,
                                        time_thre,
                                        dis_thre, 
                                        speed_thre)
    track_sets = GetJumpTrackSets(df_count.wifi_a.values,df_count.wifi_b.values)


    _STATUS_ = [False]*len(track_sets)#记录该mac下每个跳动探针组的激活状态

    status_light_list = []#记录每个跳动探针组每个探针的出现状态，当一个探针组中的每个探针都被点亮时，该探针组被激活

    actived_lights_list = []# 用于检测是否在当前状态，当lights中各值为1时，other_count会被清零

    other_count_list = [] #当当前status的other_count>3时退出当前status的激活状态

    virtual_track_list = []#记录所有status对应的virtual track

    for set in track_sets:
        status_light_list.append(dict(zip(set,[0]*len(set))))
        virtual_track_list.append(utils.GetVirtualTrack(df_wifipos,set))
        actived_lights_list.append(dict(zip(set,[0]*len(set))))
        other_count_list.append(0)

    def CheckActiveState(index):
        #check active state
        for i in range(len(status_light_list)):
            if 0 not in status_light_list[i] and _STATUS_[i] == False:
                # new status activate
                activeSet_now = track_sets[i]
                _STATUS_[i] = True
                #backward at most 5 datas to replace active tracks by virtual track
                for j in range(5):
                    index_now = index - j
                    if index_now < 0:
                        break
                    if df_now.iloc[index_now].a in activeSet_now:
                        df_now.at[index_now,'a'] = virtual_track_list[i]

    #当status[i] = len(track_sets[i])时，激活track_sets[i]
    for index,row in df_now.iterrows():
        
        #record status
        for i in range(len(track_sets)):
            if row.a in track_sets[i]:
                _STATUS_[i] += 1
        CheckActiveState(index)
        
        # if row now is active row
        replaced = False
        for i in range(len(_STATUS_)):
            if _STATUS_[i] == False:
                continue
            if row.a in track_sets[i]:
                if not replaced:
                    df_now.at[index,'a'] = virtual_track_list[i]
                    replaced = True
                
                actived_lights_list[i][row.a] = 1
                #all lights on?
                if 0 not in actived_lights_list[i].values():
                    other_count_list[i] = 0
            else:
                other_count_list[i] += 1
                if other_count_list[i] > 5:
                    _STATUS_[i] = False

    return utils.DeleteRepeatTrack(df_now)

def AddTrackCoupleToDf(df,df_wifipos,wifi_a,wifi_b,switch_t,switch_speed):
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
        dis = utils.GetWifiTrackDistance(wifi_a,wifi_b,df_wifipos)
        df = df._append({'wifi_a':wifi_a,'wifi_b':wifi_b,'count':1,'distance':dis,'meanTime':switch_t,'maxSpeed':switch_speed},ignore_index = True)
    return df

def GetJumpWifiTrackCouple(df_now,df_wifipos,count_thre = 13,time_thre = 300,dis_thre = 89,speed_thre = 26):
    #get track switch count
    last_track = 0
    df_count = pd.DataFrame({'wifi_a':[],'wifi_b':[],'count':[]})

    for index,row in df_now.iterrows():
        if last_track == 0:
            last_track = row.a
            continue
        if last_track != row.a:
            t = df_now.iloc[index].t-df_now.iloc[index - 1].t 
            dis = utils.GetWifiTrackDistance(df_now.iloc[index].a,df_now.iloc[index-1].a,df_wifipos)
            seconds = t.total_seconds() if t.total_seconds() > 0 else 0.5
            speed = dis/seconds
            df_count = AddTrackCoupleToDf(df_count,df_wifipos,last_track,row.a,t,speed)
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

def AddNewWifiTrack(df_wifiposNew,jumpTrack_sets,label):
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
        df_wifiposNew = df_wifiposNew._append({'wifi':newTrack,'X':xx,'Y':yy,'parents':info,'ID':label},ignore_index=True)
    return df_wifiposNew

def InsightTrack(df,df_pos):
    df_insight = pd.DataFrame({'wifi_a':[],'wifi_b':[],'count':[],'distance':[],'meanTime':[],'maxSpeed':[]})
    mac_list = df.m.unique()
    for mac in tqdm(mac_list,desc="评估数据"):
        df_now = utils.GetDfNow(df,mac)
        
        last_track = 0
        #df_once = GetFirstTrack(df_now)
        df_count_now = pd.DataFrame({'wifi_a':[],'wifi_b':[],'count':[],'distance':[],'meanTime':[],'maxSpeed':[]})

        #get all switch info to df_count
        for index,row in df_now.iterrows():
            if last_track == 0:
                last_track = row.a
                continue
            if last_track !=row.a:
                t = df_now.iloc[index].t-df_now.iloc[index - 1].t 
                dis = utils.GetWifiTrackDistance(df_now.iloc[index].a,df_now.iloc[index-1].a,df_pos)
                seconds = t.total_seconds() if t.total_seconds() > 0 else 0.5
                speed = dis/seconds
                df_count_now = AddTrackCoupleToDf(df_count_now,df_pos,last_track,row.a,t,speed)
                last_track = row.a

        #concat df_count
        df_insight = pd.concat([df_insight,df_count_now],axis=0)

    df_insight.meanTime = df_insight.meanTime.apply(lambda x :x.total_seconds())
    df_insight.meanTime = df_insight.meanTime/df_insight['count']
    return df_insight

def GenerateVirtualTracker(df,df_wifipos,count_thre = 13,time_thre = 300,dis_thre = 89, speed_thre = 26,label = 'virtual'):
    df_wifiposNew = df_wifipos.copy()
    mac_list = df.m.unique()
    for mac in tqdm(mac_list,desc='生成虚拟探针'):
    #for i in range(1):
        #get df now
        df_now = utils.GetDfNow(df,mac)
        
        df_couple = GetJumpWifiTrackCouple(df_now,df_wifiposNew,count_thre,time_thre,dis_thre,speed_thre)
        if len(df_couple) == 0:
            continue
        
        #get jump track sets
        track_sets = GetJumpTrackSets(df_couple.wifi_a.values,df_couple.wifi_b.values)
        #add new virtual tracks
        df_wifiposNew = AddNewWifiTrack(df_wifiposNew,track_sets,label)

    df_wifiposNew['restored_x'] = [-1]*len(df_wifiposNew)
    df_wifiposNew['restored_y'] = [-1]*len(df_wifiposNew)
    for i,row in df_wifiposNew.iterrows():
        if row.ID == 'real':
            df_wifiposNew.at[i,'restored_x'] = row.X
            df_wifiposNew.at[i,'restored_y'] = row.Y

    return df_wifiposNew