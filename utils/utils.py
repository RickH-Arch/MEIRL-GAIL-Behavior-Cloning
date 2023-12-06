import numpy as np
import sys
sys.path.append('../plot/')
import myplot
from collections import namedtuple
import pandas as pd

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


def AddCoupleToDf(df_couple_count,wifi_a,wifi_b):
    '''
    Add new couple if not exist, increace count if exist
    '''
    get = False
    df = df_couple_count.copy()
    for index,row in df.iterrows():
        if (row['wifi_a'] == wifi_a and row['wifi_b'] == wifi_b) or (row['wifi_a'] == wifi_b and row['wifi_b'] == wifi_a):
            get = True
            df.at[index,'count'] += 1
            break
    if get == False:
        df = df._append({'wifi_a':wifi_a,'wifi_b':wifi_b,'count':1},ignore_index = True)
    return df