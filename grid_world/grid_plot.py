import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_white'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import os
from utils import utils
from tqdm import tqdm
import pandas as pd
from datetime import datetime
current_time = datetime.now()
date = str(current_time.month)+str(current_time.day)

def ShowGridWorld(grid):
    fig = go.Figure(data=go.Heatmap(
                    z=grid,))

    fig.update_layout(
        title='Grid World',
        autosize=False,
        width=600,
        height=450,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    fig.show()

def ShowGridWorlds(grids_dict):
    fig = make_subplots(rows=float.__ceil__(len(grids_dict)/4),
                        cols=4,
                        subplot_titles=list(grids_dict.keys()))
    for i,grid in enumerate(grids_dict.values()):
        row_loc = int(i/4)+1
        col_loc = i%4+1
        fig.add_trace(go.Heatmap(z=grid),row=row_loc,col=col_loc)
    fig.update_layout(
        title='Grid World',
        autosize=False,
        width=300 * (len(grids_dict) if len(grids_dict)<3 else 3),
        height=160*float.__ceil__(len(grids_dict)/3),
        margin=dict(l=30, r=30, b=30, t=30)
    )
    fig.show()
    