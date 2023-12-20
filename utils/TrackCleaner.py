import numpy as np
import pandas as pd
import random
import math
import os
from utils import utils
from tqdm import tqdm
from utils.BoxFeatures import BoxFeature


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

    return ReplaceJumpTrack(df_now,df_wifipos,track_sets)

def JumpTrackRestoreWithTrackList(df_now,df_wifipos,track_sets_enforce,track_sets):
    if len(track_sets_enforce)>0:
        df_now = ReplaceJumpTrack(df_now,df_wifipos,track_sets_enforce,enforce=True)
    if len(track_sets)>0:
        df_now = ReplaceJumpTrack(df_now,df_wifipos,track_sets,enforce=False)
    return df_now


def ReplaceJumpTrack(df_now,df_wifipos,track_sets,enforce = False):
    '''
    when enforce == True, status won't be deactivated from the time it be activated
    '''

    _STATUS_ = [0]*len(track_sets)#记录该mac下每个跳动探针组的激活状态

    status_light_list = []#记录每个跳动探针组每个探针的出现状态，当一个探针组中的每个探针都被点亮时，该探针组被激活

    absent_count_list = [] #当属于该状态的任意一个探针连续缺席超过5次时，退出当前激活状态

    virtual_track_list = []#记录所有status对应的virtual track

    for set in track_sets:
        status_light_list.append(dict(zip(set,[0]*len(set))))
        absent_count_list.append(dict(zip(set,[0]*len(set))))
        virtual_track_list.append(utils.GetVirtualTrack(df_wifipos,set))

    def CheckActiveState(index):
        #check active state
        for i in range(len(status_light_list)):
            if 0 not in status_light_list[i].values():
                # new status activate
                
                AcitvateState(index,i)
                

    def AcitvateState(row_index,state_index):
        activeSet_now = track_sets[state_index]
        # reset status light
        status_light_list[state_index] = dict(zip(track_sets[state_index],[0]*len(track_sets[state_index])))
        # activate state
        if len(status_light_list[state_index].values()) == 2:
            _STATUS_[state_index] += 1
        else:
            #triangle state activate
            _STATUS_[state_index] += 100
        
        InitState(state_index)
        if enforce == False or row_index>200:
            #backward at most 5 datas to replace active tracks by virtual track
            for j in range(5):
                index_now = row_index - j
                if index_now < 0:
                    break
                if df_now.iloc[index_now].a in activeSet_now:
                    df_now.at[index_now,'a'] = virtual_track_list[state_index]
                    
        else:
            for j in range(row_index):
                ind_now = row_index -j
                EnforceReplaceTrack(ind_now)


    def InitState(state_index):
        absent_count_list[state_index] = dict(zip(track_sets[state_index],[0]*len(track_sets[state_index])))

    def CheckDeactivateState():
        for state_index in range(len(_STATUS_)):
            if max(list(absent_count_list[state_index].values()))>5:
                DeactivateState(state_index)

    def DeactivateState(state_index):
        if enforce == True:
            return
        status_light_list[state_index] = dict(zip(track_sets[state_index],[0]*len(track_sets[state_index])))
        _STATUS_[state_index] = 0

    def EnforceReplaceTrack(row_index):
        for i in range(len(_STATUS_)):
            if _STATUS_[i] == 0:
                continue
            if df_now.iloc[row_index].a in track_sets[i]:
                if _STATUS_[i] == max(_STATUS_):
                    df_now.at[row_index,'a'] = virtual_track_list[i]
                    return
    
    t1 = df_now.iloc[0].t
    time_delta1 = 0
    lock_list = []
    #当status[i] = len(track_sets[i])时，激活track_sets[i]
    for index,row in df_now.iterrows():

        #deactivate all state if time_delta1 > 30min
        time_delta2 = row.t - t1
        
        if time_delta2.total_seconds() > 1800 and enforce == False:
            for i in range(len(_STATUS_)):
                DeactivateState(i)

        #lock data if row now is time isolated
        if time_delta1 != 0:
            if time_delta1> pd.Timedelta('30min') and time_delta2>pd.Timedelta('30min'):
                lock_list.append({index:row.a})

        t1 = row.t
        time_delta1 = time_delta2

        #record status
        for i in range(len(track_sets)):
            if row.a in track_sets[i]:
                status_light_list[i][row.a] += 1 
        CheckActiveState(index)
        
        # if row now is active row
        
        for i in range(len(_STATUS_)):
            if _STATUS_[i] == 0:
                continue
            changed = False
            if row.a in track_sets[i]:
                if _STATUS_[i] == max(_STATUS_):
                    changed = True
                    df_now.at[index,'a'] = virtual_track_list[i]
                if enforce and changed == False:
                    df_now.at[index,'a'] = virtual_track_list[i]

                
                #any absent tracker?
                keys = list(absent_count_list[i].keys())
                for key in keys:
                    if key != row.a:
                        absent_count_list[i][key] += 1
                    else:
                        absent_count_list[i][key] = 0
            
        CheckDeactivateState()

    #restore locked data   
    for lock in lock_list:
        for key,value in lock.items():
            df_now.at[key,'a'] = value

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
        new_set = set([a,b])
        track_sets.append(new_set)
        #find if there are potential triangle set
        for track_set in track_sets:
            if a in track_set and len(track_set) == 2:
                c = _getTrackSetAnotherTrack(track_set,a)
                if c == b:
                    continue
                for other_set in track_sets:
                    if other_set == track_set:
                        continue
                    if c in other_set and b in other_set:
                        #append new triangle
                        track_sets.append(set([a,b,c]))
                        break

    return track_sets

def GetEnforceJumpTrackSets(track_list1,track_list2):
    if len(track_list1) != len(track_list2):
        return
    track_sets = []
    for i in range(len(track_list1)):
        a = int(track_list1[i])
        b = int(track_list2[i])
        added = False
        for track_set in track_sets:
        #find if there are potential triangle set
            if a in track_set and len(track_set) == 2:
                c = _getTrackSetAnotherTrack(track_set,a)
                for other_set in track_sets:
                    if c in other_set and b in other_set and len(other_set) == 2:
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
        while newTrack in df_wifiposNew.wifi or newTrack<1000:
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

def GenerateVirtualTrackerReturnTrackList(df,
                                          df_wifipos,
                                          enforce_count_thre,
                                          count_thre,
                                          time_thre,
                                          dis_thre, 
                                          speed_thre,
                                          label = 'virtual'):
    '''
    return[0]:总共新创建的探针列表
    return[1]:每个mac下创建虚拟探针的父探针set集合
    '''
    df = df.copy()
    df_wifiposNew = df_wifipos.copy()
    mac_list = df.m.unique()
    
    normalSets_list = [[]]*len(mac_list)
    enforceSets_list = [[]]*len(mac_list)
    i = -1
    for mac in tqdm(mac_list,desc='生成虚拟探针'):
        i+=1
        df_now = utils.GetDfNow(df,mac)

        #获取跳动探针对
        df_couple = GetJumpWifiTrackCouple(df_now,df_wifiposNew,count_thre,time_thre,dis_thre,speed_thre)
        if len(df_couple) == 0:
            continue

        #生成跳动探针组
        df_couple_enforce = df_couple[df_couple['count']>=enforce_count_thre]
        df_couple = df_couple[df_couple['count']<enforce_count_thre]
        track_sets_enforce = GetEnforceJumpTrackSets(df_couple_enforce.wifi_a.values,df_couple_enforce.wifi_b.values)
        track_sets_normal = GetJumpTrackSets(df_couple.wifi_a.values,df_couple.wifi_b.values)
        track_sets_all = track_sets_normal+track_sets_enforce
        
        normalSets_list[i] = track_sets_normal
        enforceSets_list[i] = track_sets_enforce

        #添加新探针到探针信息列表
        df_wifiposNew = AddNewWifiTrack(df_wifiposNew,track_sets_all,label)
        

    df_wifiposNew['restored_x'] = [-1]*len(df_wifiposNew)
    df_wifiposNew['restored_y'] = [-1]*len(df_wifiposNew)
    for i,row in df_wifiposNew.iterrows():
        if row.ID == 'real':
            df_wifiposNew.at[i,'restored_x'] = row.X
            df_wifiposNew.at[i,'restored_y'] = row.Y

    return df_wifiposNew,normalSets_list,enforceSets_list

def GenerateVirtualTracker(df,df_wifipos,count_thre = 13,time_thre = 300,dis_thre = 89, speed_thre = 26,label = 'virtual'):
    '''
    return[0]:总共新创建的探针列表
    '''
    df_wifiposNew = df_wifipos.copy()
    mac_list = df.m.unique()
    for mac in tqdm(mac_list,desc='生成虚拟探针'):
        df_now = utils.GetDfNow(df,mac)
        df_couple = GetJumpWifiTrackCouple(df_now,df_wifiposNew,count_thre,time_thre,dis_thre,speed_thre)
        if len(df_couple) == 0:
            continue
        
        #get jump track sets
        track_sets = GetJumpTrackSets(df_couple.wifi_a.values,df_couple.wifi_b.values)
        #add new virtual tracks
        df_wifiposNew = AddNewWifiTrack(df_wifiposNew,track_sets,label)
        i+=1

    df_wifiposNew['restored_x'] = [-1]*len(df_wifiposNew)
    df_wifiposNew['restored_y'] = [-1]*len(df_wifiposNew)
    for i,row in df_wifiposNew.iterrows():
        if row.ID == 'real':
            df_wifiposNew.at[i,'restored_x'] = row.X
            df_wifiposNew.at[i,'restored_y'] = row.Y

    return df_wifiposNew

def AutoGetThreshold(df_insight):
    '''
    return[0]:enforce_count_thre
    return[1]:count_thre
    return[2]:time_thre
    return[3]:dis_thre
    return[4]:speed_thre
    '''
    #获取强制转换切换次数临界值
    enforce_count_thre = BoxFeature(df_insight['count'])[5] #upper fence
    if enforce_count_thre < 50:
        enforce_count_thre = 50

    #获取切换次数临界值 - Q3
    count_thre = BoxFeature(df_insight['count'])[4]
    if count_thre < 8:
        count_thre = 8

    #获取平均切换时间 - Q3+50
    
    time_thre = BoxFeature(df_insight['meanTime'])[4]+50

    #获取满足次数下的平均切换距离 - count>count_thre -> Q3
    
    distance_thre = BoxFeature(df_insight[df_insight['count']>count_thre]['distance'])[4]

    #获取满足次数下的最大切换速度 - count>count_thre -> Q1
    
    speed_thre = BoxFeature(df_insight[df_insight['count']>count_thre]['maxSpeed'])[2]

    return enforce_count_thre,count_thre,time_thre,distance_thre,speed_thre