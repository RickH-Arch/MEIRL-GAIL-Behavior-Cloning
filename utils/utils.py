import numpy as np

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