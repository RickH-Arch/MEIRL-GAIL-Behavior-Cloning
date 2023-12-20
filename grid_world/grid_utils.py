import pandas as pd
import numpy as np
from utils import utils


def TrackToGridPoints(df_now,df_wifipos,df_path):
    x,y,z = utils.GetPathPoints(df_now,df_wifipos,df_path)