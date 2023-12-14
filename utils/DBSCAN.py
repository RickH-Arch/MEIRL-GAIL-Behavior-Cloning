from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score # 轮廓系数评估函数
from sklearn.metrics import calinski_harabasz_score #卡林斯基-哈拉巴斯指数
from collections import namedtuple
import numpy as np
from utils.DBCV import DBCV
from tqdm import tqdm

DBSCAN_RESULT = namedtuple('dbscan_result',
                           ['cluster_num',
                            'noise_num',
                            'dbcv',
                            'silhouette',
                            'calinski',
                            'eps',
                            'min_samples',
                            'labels'])

#剔除dbscan的噪声（即值为-1）
def _eliminateNoise(df,labels):
    df_result = df.copy()
    df_result['label'] = labels
    df_result = df_result[df_result.label != -1]
    return df_result

def My_DBSCAN(df_cluster,eps,min_samples,cal_dbcv = False):
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster = dbscan.fit(df_cluster)
    df_result = _eliminateNoise(df_cluster,cluster.labels_)

    #create array cluster for dbcv evaluation
    arr_cluster = []
    for index,row in df_cluster.iterrows():
        arr_cluster.append([row[0],row[1],row[2]])
    arr_cluster = np.array(arr_cluster)

    #get labels
    labels = []
    for k in range(len(cluster.labels_)):
        labels.append(cluster.labels_[k])
    cluster_labels = labels

    if len(set(cluster.labels_)) <= 2:
        cluster_n = len(set(cluster.labels_)) - (1 if -1 in cluster.labels_ else 0)
        
        silhouette = -1
        calinski = -1
        noise_n = 0
        dbcv=-1
    else:
        cluster_n = (len(set(cluster.labels_)) - (1 if -1 in cluster.labels_ else 0))#delete npise
        #get noise num
        noise_n = np.count_nonzero(cluster.labels_ == -1)

        #--get evaluations--#
        silhouette = silhouette_score(df_result, df_result.label)
        calinski = calinski_harabasz_score(df_result, df_result.label)
        #DBCV
        labels = np.array(df_result.label)
        arr_cluster_now = arr_cluster[np.where(cluster.labels_ != -1)]
        if(cal_dbcv):
            dbcv = DBCV(arr_cluster_now,labels)
        else:
            dbcv = -1

    return DBSCAN_RESULT(
        cluster_num=cluster_n,
        noise_num = noise_n,
        dbcv = dbcv,
        silhouette=silhouette,
        calinski = calinski,
        eps = eps,
        min_samples=min_samples,
        labels = cluster_labels
    )


def My_DBSCAN_MATRIX(df_cluster,init_eps,step_eps,epoch_eps,init_mSamples,step_mSamples,epoch_mSamples,cal_DBCV = False):
    '''
    多次聚类, 确定效果最佳的eps与min_samples
    '''
    results = []
    for i in tqdm(range(epoch_eps),desc='on dbscan...'):
        for j in (range(epoch_mSamples)):
        #eps表示样本点的领域半径，min_samples表示样本点在领域半径内的最小数量
            result = My_DBSCAN(df_cluster,init_eps+i*step_eps,init_mSamples+j*step_mSamples,cal_DBCV)
            results.append(result) 
    return results