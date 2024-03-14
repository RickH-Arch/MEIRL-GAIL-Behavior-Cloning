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

import math

import plotly.io as pio

try:
    img_path = os.getcwd()+'/wifi_track_data/dacang/imgs/roads.png'
    img = Image.open(img_path)
    background_img = img
    buttom_img = Image.fromarray(np.array(img.transpose(Image.FLIP_TOP_BOTTOM))).convert('P', palette='WEB', dither=None)
except:
    print("no background image")
    background_img = Image.fromarray(np.ones((300,400,3), dtype='uint8')).convert('P', palette='WEB')
    buttom_img = Image.fromarray(np.ones((300,400,3), dtype='uint8')).convert('P', palette='WEB')

dum_img = Image.fromarray(np.ones((3,3,3), dtype='uint8')).convert('P', palette='WEB')
idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))
colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
im_x = np.linspace(0, 400, 400)
im_y = np.linspace(0, 300, 300)
im_z = np.zeros((300,400))

colors = ['rgb(67,67,67)', 'rgb(115,115,115)']
line_size = [2,2,2,2]

def ShowGridWorld(grid,width = 600,height = 450,title = "Grid World"):
    fig = go.Figure(data=go.Heatmap(
                    z=grid,))

    fig.update_layout(
        title=title,
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=20, r=20, b=50, t=50),
        
    )
    fig.show()

def ShowGridWorld_anime(grids,width = 600,height = 450,title = "Grid World"):
    fig = go.Figure(
        data=[go.Heatmap(z=grids[0],)],
        layout=go.Layout(
            title=title,
            autosize=False,
            width=width,
            height=height,
            margin=dict(l=20, r=20, b=50, t=50),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": 10, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 500}}]),
                         dict(label="Pause",
                              method="animate",
                              args=[[None], {"frame": {"duration": 0, "redraw": False},
                                             "mode": "immediate",
                                             "transition": {"duration": 0}}])])]
        ),
         frames=[go.Frame(data=go.Heatmap(z=grid)) for grid in grids],   
        )
        
    fig.show()

def barchart3d(labels, z_data, title, z_title,
               n_row=0, width=900, height=900, thikness=0.7, colorscale='Viridis',
               **kwargs):
    """
    Draws a 3D barchart
    :param labels: Array_like of bar labels
    :param z_data: Array_like of bar heights (data coords)
    :param title: Chart title
    :param z_title: Z-axis title
    :param n_row: Number of x-rows
    :param width: Chart width (px)
    :param height: Chart height (px)
    :param thikness: Bar thikness (0; 1)
    :param colorscale: Barchart colorscale
    :param **kwargs: Passed to Mesh3d()
    :return: 3D barchart figure
    """

    if n_row < 1:
        n_row = math.ceil(math.sqrt(len(z_data)))
    thikness *= 0.5
    ann = []
    
    fig = go.Figure()

    for iz, z_max in enumerate(z_data):
        x_cnt, y_cnt = iz % n_row, iz // n_row
        x_min, y_min = x_cnt - thikness, y_cnt - thikness
        x_max, y_max = x_cnt + thikness, y_cnt + thikness

        fig.add_trace(go.Mesh3d(
            x=[x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max],
            y=[y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min],
            z=[0, 0, 0, 0, z_max, z_max, z_max, z_max],
            alphahull=0,
            intensity=[0, 0, 0, 0, z_max, z_max, z_max, z_max],
            coloraxis='coloraxis',
            hoverinfo='skip',
            **kwargs))

        ann.append(dict(
            showarrow=False,
            x=x_cnt, y=y_cnt, z=z_max,
            text=f'<b>#{iz+1}</b>',
            font=dict(color='white', size=11),
            bgcolor='rgba(0, 0, 0, 0.3)',
            xanchor='center', yanchor='middle',
            hovertext=f'{z_max} {labels[iz]}'))
   
    # mesh3d doesn't currently support showLegend param, so
    # add invisible scatter3d with names to show legend
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            opacity=0,
            name=f'#{i+1} {label}'))

    fig.update_layout(
        width=width, height=height,
        title=title, title_x=0.5,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(title=z_title),
            annotations=ann),
        coloraxis=dict(
            colorscale=colorscale,
            colorbar=dict(
                title=dict(
                    text=z_title,
                    side='right'),
                xanchor='right', x=1.0,
                xpad=0,
                ticks='inside')),
        legend=dict(
            yanchor='top', y=1.0,
            xanchor='left', x=0.0,
            bgcolor='rgba(0, 0, 0, 0)',
            itemclick=False,
            itemdoubleclick=False),
        showlegend=True)
    return fig

def BarChart_3D(coords,z_data,thikness=3,width=600,height=600,title='BarChart 3D',z_title='probability',bar_colorscale='Viridis'):
    ann = []
    fig = go.Figure()

    for iz, z_max in enumerate(z_data):
        
        x_coord = coords[iz][0]
        y_coord = coords[iz][1]
        x_min, y_min = x_coord - thikness, y_coord - thikness
        x_max, y_max = x_coord + thikness, y_coord + thikness

        fig.add_trace(go.Mesh3d(
            x=[x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max],
            y=[y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min],
            z=[0, 0, 0, 0, z_max, z_max, z_max, z_max],
            alphahull=0,
            intensity=[0, 0, 0, 0, z_max, z_max, z_max, z_max],
            coloraxis='coloraxis',
            hoverinfo='skip',
            ))
        
    #add buttom background image
    fig.add_trace(go.Surface(x=im_x, y=im_y, z=im_z,
        surfacecolor=buttom_img, 
        cmin=0, 
        cmax=255,
        colorscale=colorscale,
        showscale=False,
        lighting_diffuse=1,
        lighting_ambient=1,
        lighting_fresnel=1,
        lighting_roughness=1,
        lighting_specular=0.5,
    ))

    fig.update_layout(
        width=width, height=height,
        title=title, title_x=0.5,
        scene=dict(
            aspectratio=dict(x=1, y=0.75, z=0.75),
            xaxis=dict(showticklabels=False,range=[0,400], title=''),
            yaxis=dict(showticklabels=False,range=[0,300], title=''),
            zaxis=dict(title=z_title),
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=-0.5,
                    y=-1,
                    z=0.6,
                )
            ),
            ),
        coloraxis=dict(
            colorscale=bar_colorscale,
            colorbar=dict(
                title=dict(
                    text=z_title,
                    side='right'),
                xanchor='right', x=1.0,
                xpad=0,
                ticks='inside')),
        legend=dict(
            yanchor='top', y=1.0,
            xanchor='left', x=0.0,
            bgcolor='rgba(0, 0, 0, 0)',
            itemclick=False,
            itemdoubleclick=False),
        showlegend=False,
        )
    fig.update_coloraxes(showscale=False)
        
    fig.show()
    

def ShowGridWorlds(grids_dict,title = ''):
    fig = make_subplots(rows=float.__ceil__(len(grids_dict)/4),
                        cols=4,
                        subplot_titles=list(grids_dict.keys()))
    for i,grid in enumerate(grids_dict.values()):
        row_loc = int(i/4)+1
        col_loc = i%4+1
        fig.add_trace(go.Heatmap(z=grid),row=row_loc,col=col_loc)
    fig.update_layout(
        title=title,
        autosize=False,
        width=240 * (len(grids_dict) if len(grids_dict)<3 else 3),
        height=120*float.__ceil__(len(grids_dict)/3),
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
        x = []
        y = []
        if t[0] == t[2] and t[1] == t[3]:
            #stay in same state
            x.append(t[0])
            y.append(t[1])
            fig.add_trace(go.Scatter(x=x,y=y,mode='markers',
                                     marker_symbol = 'circle',
                                     marker_color = "red",
                                     marker_size=t[4]*5,))
            continue
        
        x.append(t[0])
        y.append(t[1])
        mid_x = (t[0]+t[2])/2
        mid_y = (t[1]+t[3])/2
        x.append(mid_x)
        y.append(mid_y)
        fig.add_trace(go.Scatter(x=x,y=y,mode='lines',
                                 line=dict(color='red', width=t[4]*10)))
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

def ShowTraj(track,width,height,title='traks'):
    fig = go.Figure()
    for t in track:
        x = []
        y = []
        
        x.append(t[0])
        y.append(t[1])
        x.append(t[2])
        y.append(t[3])
        fig.add_trace(go.Scatter(x=x,y=y,mode='lines',
                                 line=dict(color='red', width=t[4]/10)))
        
    fig.update_layout(
        title=title,
        autosize=False,
        width=450,
        height=450,
        xaxis = dict(range=[0,width],
                     showgrid = False),
        yaxis = dict(range=[0,height],
                     showgrid = False),
        showlegend=False,
        margin=dict(l=50, r=50, b=50, t=50),
    )
    fig.show()

def PrintTraj3D(x,y,z,x_name = "",y_name = "",z_name = "",marker_size = 3,line_width = 3,save_path=''):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
    x=x, 
    y=y, 
    z=z,
    
    marker=dict(
        color=z,
        colorscale='Viridis',
        size=marker_size,
    ),
    line=dict(
        color='rgba(50,50,50,0.6)',
        width=line_width,
        
    )
    ))

    #add buttom background image
    fig.add_trace(go.Surface(x=im_x, y=im_y, z=im_z,
        surfacecolor=buttom_img, 
        cmin=0, 
        cmax=255,
        colorscale=colorscale,
        showscale=False,
        lighting_diffuse=1,
        lighting_ambient=1,
        lighting_fresnel=1,
        lighting_roughness=1,
        lighting_specular=0.5,
    ))

    fig.update_layout(
        width=300,
        height=300,
        autosize=True,
        scene=dict(
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=0,
                    y=-1,
                    z=1,
                )
            ),
            xaxis_visible=True,
                yaxis_visible=True, 
                zaxis_visible=True, 
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="hour" ,
            aspectmode = 'manual',
            aspectratio=dict(x=1, y=0.75, z=0.75),
            xaxis = dict(nticks=4, range=[0,40],),
            yaxis = dict(nticks=4, range=[0,30],),
            zaxis = dict(nticks=4, ),
        ),
    )


    # if os.path.exists(save_path) == False:
    #     os.makedirs(save_path)
    #pio.write_image(fig,'traj.png')
    fig.show()




    
