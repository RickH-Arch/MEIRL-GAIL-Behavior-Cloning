import pandas as pd
from grid_world.grid_world import GridWorld
from DMEIRL.DeepMEIRL_FC import DMEIRL
from utils import utils

#------------------------------------Initialize Grid World------------------------------------------

env_folder_path = 'wifi_track_data/dacang/grid_data/env_imgs/40_30'
expert_traj_path = "wifi_track_data/dacang/track_data/trajs_sliced_0229_40x30.csv"

#env_folder_path = "wifi_track_data/dacang/grid_data/envs_grid/0117_40x30"
#feature_folder_path = "wifi_track_data/dacang/grid_data/features_grid/0117_40x30"


world = GridWorld(
                  expert_traj_filePath=expert_traj_path,
                  environments_img_folderPath=env_folder_path,
                  width=40, height=30,discount=0.95,trans_prob=0.8)
# df_cluster = pd.read_csv('wifi_track_data/dacang/cluster_data/cluster_result_0203.csv')
# world.experts.ReadCluster(df_cluster)
# world.experts.ApplyCluster((0,1,2))
print("GridWorld initialized")

#------------------------------------Initialize DMEIRL------------------------------------------

dme = DMEIRL(world,layers=(60,120,240,120,60),lr=0.0001,weight_decay=0.2,log=f'{utils.date}sliced_bias{world.traj_len_bias}_v0.001_tp{world.trans_prob}_dis{world.discount}',log_dir='run_sliced')

#------------------------------------Train------------------------------------------

dme.train(n_epochs=500)

# layers_list = [(60,120,60),(60,240,60),(120,240,120),(60,120,240,120,60),(60,120,240,240,120,60)]
# wd_list = [0.2,0.5,1]
# bias_list = [0,10,15,20]

# total_count = len(layers_list)*len(wd_list)*len(bias_list)
# count = 1
# for bias in bias_list:
#     world.experts.ChangeTrajLenBias(bias)
#     for wd in wd_list:
#         for l in layers_list:
#             dme = DMEIRL(world,layers=l,lr=0.0001,weight_decay=wd,log=f'{utils.date}sliced_bias{world.experts.bias}_v0.001_tp{world.trans_prob}_dis{world.discount}',log_dir='run_sliced')
#             print(f"----------------{dme.info}----------------")
#             print(f"----------------start{count}/{total_count}----------------")
#             dme.train(n_epochs=400,demo=True,save=True)
#             count+=1

