import pandas as pd
from grid_world.grid_world import GridWorld
from DMEIRL.DeepMEIRL_FC import DMEIRL

#------------------------------------Initialize Grid World------------------------------------------

env_folder_path = r'wifi_track_data/dacang/grid_data/env_imgs/40_30'
expert_traj_path = r"wifi_track_data\dacang\track_data\trajs_0117_40x30.csv"

world = GridWorld(environments_img_folderPath=env_folder_path,
                  expert_traj_filePath=expert_traj_path,
                  width=40, height=30)
df_cluster = pd.read_csv('wifi_track_data/dacang/cluster_data/cluster_result_0203.csv')
world.experts.ReadCluster(df_cluster)
world.experts.ApplyCluster((0,1,2))
print("GridWorld initialized")

#------------------------------------Initialize DMEIRL------------------------------------------

dme = DMEIRL(world,layers=(30,60,30),lr=0.0005,weight_decay=0.5,log='0123',log_dir='run')

#------------------------------------Train------------------------------------------

dme.train(n_epochs=5000)