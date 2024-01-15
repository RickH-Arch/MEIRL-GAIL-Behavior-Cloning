import pandas as pd
from grid_world.grid_world import GridWorld
from DMEIRL.DeepMEIRL_FC import DMEIRL

#------------------------------------Initialize Grid World------------------------------------------

env_folder_path = "wifi_track_data/dacanggrid_data/envs_grid/1230"
feature_folder_path = "wifi_track_data/dacang/grid_data/features_grid/1230"
expert_traj_path = "wifi_track_data/dacang/track_data/trajs_1230.csv"

world = GridWorld(features_folderPath=feature_folder_path,
                  expert_traj_filePath=expert_traj_path,
                  width=80, height=60)
df_cluster = pd.read_csv('wifi_track_data/dacang/cluster_data/cluster_result_0105.csv')
world.experts.ReadCluster(df_cluster)
world.experts.ApplyCluster((0,1,2))
print("GridWorld initialized")

#------------------------------------Initialize DMEIRL------------------------------------------

dme = DMEIRL(world,layers=(60,30,30),lr=0.0001,clip_norm=0.1,weight_decay=0.1,log='0115')

#------------------------------------Train------------------------------------------

dme.train(n_epochs=10000)