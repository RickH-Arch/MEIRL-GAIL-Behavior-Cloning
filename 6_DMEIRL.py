import pandas as pd
import numpy as np
import os
import datetime
import time
from tqdm import tqdm

import math

from utils import utils,myplot
from grid_world.grid_world import GridWorld
from DMEIRL.DeepMEIRL_FC import DMEIRL

import warnings
warnings.filterwarnings('ignore')

#------------------------------------Initialize Grid World------------------------------------------

count_path = r"wifi_track_data\dacang\grid_data\count_grid_1221.npy"
env_folder_path = r"wifi_track_data\dacang\grid_data\envs_grid\1223"
feature_folder_path = r"wifi_track_data\dacang\grid_data\features_grid\1223"
expert_traj_path = r"wifi_track_data\dacang\track_data\trajs_1221.csv"

world = GridWorld(count_path,env_folder_path,feature_folder_path,expert_traj_path)
print("GridWorld initialized")

#------------------------------------Initialize DMEIRL------------------------------------------

dme = DMEIRL(world,layers=(40,30))

#------------------------------------Train------------------------------------------

dme.train(n_epochs=3000)