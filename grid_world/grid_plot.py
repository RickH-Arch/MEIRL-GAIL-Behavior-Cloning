import plotly.express as px
import plotly.io as pio
pio.templates.default = 'plotly_white'
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import os
from utils import utils
from grid_world import grid_utils
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
        margin=dict(l=65, r=50, b=65, t=90),
        
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

def ShowDynamics(dynamic_track,dir,width,height,grid):

    
    track = dynamic_track[dir]
    fig = go.Figure()
    fig=go.Figure(data=go.Heatmap(
                    z=grid,
                    colorscale="Mint",
                    showscale=False))

    origin_p = [[],[]]
    for w in range(width):
        for h in range(height):
            origin_p[0].append(w)
            origin_p[1].append(h)

    fig.add_trace(go.Scatter(x=origin_p[0],y=origin_p[1],mode='markers',
                             marker_symbol = 'square-open',
                             marker_line_width=0.2,
                             marker_line_color = "lightgray",
                             opacity=0.3,
                             marker_size=7,))

    for t in track:
        if t[0] == t[2] and t[1] == t[3]:
            continue
        x = []
        y = []
        x.append(t[0])
        y.append(t[1])
        mid_x = (t[0]+t[2])/2
        mid_y = (t[1]+t[3])/2
        x.append(mid_x)
        y.append(mid_y)
        fig.add_trace(go.Scatter(x=x,y=y,mode='lines',
                                 line=dict(color='red', width=t[4]*5)))
    fig.update_layout(
        title='Dynamics',
        autosize=False,
        width=600,
        height=450,
        xaxis = dict(range=[0,width],
                     showgrid = False),
        yaxis = dict(range=[0,height],
                     showgrid = False),
        showlegend=False,
        margin=dict(l=10, r=10, b=10, t=10),
    )

    fig.show()
    
        




    
