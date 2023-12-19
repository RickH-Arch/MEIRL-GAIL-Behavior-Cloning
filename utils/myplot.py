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


img_path = os.getcwd()+'/wifi_track_data/dacang/imgs/roads.png'
img = Image.open(img_path)
background_img = img
buttom_img = Image.fromarray(np.array(img.transpose(Image.FLIP_TOP_BOTTOM))).convert('P', palette='WEB', dither=None)
dum_img = Image.fromarray(np.ones((3,3,3), dtype='uint8')).convert('P', palette='WEB')
idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))
colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
im_x = np.linspace(0, 400, 400)
im_y = np.linspace(0, 300, 300)
im_z = np.zeros((300,400))

colors = ['rgb(67,67,67)', 'rgb(115,115,115)']
line_size = [2,2,2,2]

def Bubble_Scatter(df,x_name,y_name,size_name,hover_name = "M",xAxes_name = '',yAxes_name = ''):
    '''
    绘制Date-Mac的气泡图
    '''
    fig = px.scatter(df,x=x_name,y=y_name,size=size_name,color="oriMac",hover_name=hover_name)
    if xAxes_name != '':
        fig.update_xaxes(title=xAxes_name)
    if yAxes_name != '':
        fig.update_yaxes(title=yAxes_name)
    fig.show()

def One_Axes_Line(df,xAxes_str,line_str,xAxes_name = "",line_name = ""):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x = df[xAxes_str], y = df[line_str],
                            mode='lines',
                            line=dict(color=colors[0],width=line_size[0]),
                            ),
                )
    fig.update_layout(
        width = 800,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
    )
    if xAxes_name != "":
        fig.update_xaxes(title_text="<b>%i</b>"%xAxes_name)
    if line_name != "":
        fig.update_yaxes(title_text="<b>%i</b>"%line_name)
    fig.show()

def Double_Axes_Line(df,xAxis_str,line1_str,line2_str,xAxes_name = "",line1_name = "",line2_name = ""):
    '''
    绘制具有两个y轴的折线图
    '''
    line_size = [2, 2]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x = df[xAxis_str], y = df[line1_str],
                            mode='lines',
                            line=dict(color=colors[0],width=line_size[0]),
                            ),
                            secondary_y = False)
    fig.add_trace(
        go.Scatter(x = df[xAxis_str], y = df[line2_str],
                            mode='lines',
                            line=dict(color=colors[1],width=line_size[1]),
                            ),
                            secondary_y = True)
    if line1_name != "":
        fig.update_yaxes(title_text="<b>%i</b> "%line1_name, secondary_y=False)
    if line2_name != "":
        fig.update_yaxes(title_text="<b>%i</b>"%line2_name, secondary_y=True)
    fig.update_layout(
        width = 800,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
    )
    if xAxes_name != "":
        fig.update_xaxes
    fig.show()

def Scatter_2D(df,x_name,y_name,label_name = '',bg_img = 0):
    fig = go.Figure()
    if label_name == '':
        fig = px.scatter(x=df[x_name], y=df[y_name])
    else:
        fig = px.scatter(x=df[x_name], y=df[y_name],color=df[label_name])
    
    if bg_img == 0:
        bg_img = background_img
        
    fig.add_layout_image(
            dict(
                source=bg_img,
                xref="x", yref="y",
                x=0, y=0,  #position of the upper left corner of the image in subplot 1,1
                sizex= 400,sizey= 300, #sizex, sizey are set by trial and error
                xanchor="left",
                yanchor="bottom",
                sizing="stretch",
                layer="below",
                opacity=0.3)
    )

    fig.update_layout(
        width=480,
        height=300,
        autosize = False,
        
        margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=4
        ),
        #yaxis_range=[0,320],
        #xaxis_range=[0,420],
        template="plotly_white",

        legend = dict(
            title = ''
        )
    )

    #fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()

def Scatter_2D_Subplot(data_tuple_list,bg_img_path = ""):
    '''
    !!!unfixed!!!
    tuple[0]为dataframe, tuple[1]为x轴列名, tuple[2]为y轴列名, tuple[3]为label列名
    '''
    name_list = []
    for i in range(len(data_tuple_list)):
        name_list.append(data_tuple_list[i][3])
    fig = make_subplots(
        rows = float.__ceil__(len(data_tuple_list)/3),
        cols = 3
    )

    img = ""
    imgs = []
    if bg_img_path != "":
        img = Image.open(bg_img_path)

    for i in range(len(data_tuple_list)):
        t = data_tuple_list[i]
        row_loc = int(i/3)+1
        col_loc = i%3+1
        df = t[0]
        fig.add_trace(go.Scatter(
            x = df[t[1]],
            y = df[t[2]],
            color = df[t[3]],
            row = row_loc, col = col_loc
        ))
        
        if bg_img_path != "":
            imgs.append(dict(
                source=img,
                    xref="x", yref="y",
                    x=0, y=0,  #position of the upper left corner of the image in subplot 1,1
                    sizex= 400,sizey= 300, #sizex, sizey are set by trial and error
                    xanchor="left",
                    yanchor="bottom",
                    sizing="stretch",
                    layer="below",
                    opacity=0.3
            ))
    fig.update_layout(
        images = imgs,
        title_text='WiFi Track Position',
        height=400*float.__ceil__(len(data_tuple_list)/3),
        width=500 * len(data_tuple_list) if len(data_tuple_list)<3 else 3
    )

    fig.show()

def Parents_2D(df,ID = "virtual"):
    if ID == "virtual":
        df_virtual = df[df.ID.apply(lambda x : x.__contains__("virtual"))]
    else:
        df_virtual = df[df.ID == ID]
    parents_sets = []
    for i in range(len(df_virtual)):
        track_list_now = df_virtual.iloc[i]['parents'].split(':')
        l_x = []
        l_y = []
        for track in track_list_now:
            row_now = df[df.wifi == int(track)].iloc[0]
            l_x.append(row_now.X)
            l_y.append(row_now.Y)
        l_x.append(l_x[0])
        l_y.append(l_y[0])
        parents_sets.append((l_x,l_y))
    fig = go.Figure()
    for set in parents_sets:
        fig.add_trace(go.Scatter(x=set[0], y=set[1],
                            line=dict(width=1),
                            showlegend=False,))
    
    fig.add_layout_image(
            dict(
                source=background_img,
                xref="x", yref="y",
                x=0, y=0,  #position of the upper left corner of the image in subplot 1,1
                sizex= 400,sizey= 300, #sizex, sizey are set by trial and error
                xanchor="left",
                yanchor="bottom",
                sizing="stretch",
                layer="below",
                opacity=0.3)
    )

    fig.update_layout(
        width=400,
        height=300,
        autosize = False,
        
        margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=4
        ),
        yaxis_range=[0,300],
        xaxis_range=[0,400],
        template="plotly_white",
        

        legend = dict(
            title = ''
        )
    )
    fig.show()

def Scatter_3D(df,x_name,y_name,z_name,species_name = "",color_name = ""):
    '''
    绘制打上时间标签并聚类后的3d scatter
    '''
    fig = px.scatter_3d(df, x=x_name, y=y_name, z=z_name,
              color=color_name, 
               opacity=0.7,
               size = 'A_count',
               )

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

def Surface3D(z_mat,x,y,x_name = "",y_name = ""):
    fig = go.Figure(data=[go.Surface(z=z_mat, x=x, y=y)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
    fig.update_layout( autosize=False,
                    scene_camera_eye=dict(x=1.87, y=-1.5, z=0.64),
                    width=500, height=500,
                    margin=dict(l=20, r=20, b=20, t=20)
    )
    if x_name != "":
         fig.update_xaxes(title_text="<b>eps</b>")
    # if y_name != "":
    #     fig.update_yaxes(title_text="<b>%i</b>"%y_name)
    fig.show()

def Surface3D_supPlot(data_tuple_list):
    """
    绘制多个3D曲面图

    Args:
        data_tuple_list (tuple): tuple[0]为value list, tuple[1]为x轴列表, tuple[2]为y轴列表, tuple[3]为图表名称
    """
    if len(data_tuple_list) == 0:
        return
    col_num = 3
    row_num = float.__ceil__(len(data_tuple_list)/col_num)
    #get specs
    list_specs = []
    for i in range(row_num):
        l = []
        for j in range(col_num):
            l.append({'type':'surface'})
        list_specs.append(l)
    #get name tuple
    name_list = []
    for i in range(len(data_tuple_list)):
        name_list.append(data_tuple_list[i][3])
    name_tuple = tuple(name_list)

    #get fig
    fig = make_subplots(
    rows=row_num, cols=col_num,
    specs=list_specs,
    subplot_titles=name_tuple
    )

    for i in range(len(data_tuple_list)):
        t = data_tuple_list[i]
        row_loc = int(i/col_num)+1
        col_loc = i%col_num+1
        fig.add_trace(
            go.Surface(x=t[1], y=t[2], z=t[0], colorscale='Viridis', showscale=False),
            row=row_loc, col=col_loc)
        fig.update_xaxes(title_text = "min_samples",row = row_loc,col=col_loc)
        fig.update_yaxes(title_text = 'eps',row = row_loc,col=col_loc)
    
    fig.update_layout(
        title_text='Cluster Result',
        height=400*row_num,
        width=1000
    )

    fig.show()

def Boxes(list_tuple,box_title = ""):
    '''
    args:
    list_tuple[0][0]:y轴数据_list, list_tuple[0][1]:数据名称
    '''
    fig = make_subplots(rows=1, cols=len(list_tuple))
    for i,tuple in enumerate(list_tuple):
        fig.add_trace(
            go.Box(y=tuple[0],
                name=tuple[1],
                marker_size=1,
                line_width=1),
        row=1, col=i+1
    )
    fig.update_traces(boxpoints='all', jitter=.2)
    fig.update_layout(height=400, width=300*len(list_tuple))
    
    if box_title == "":
        fig.update_layout(title_text="Box Plot")
    else :
        fig.update_layout(title_text=box_title)
    
    fig.show()


def Track_3D(x,y,z,x_name = "",y_name = "",z_name = "",marker_size = 3,line_width = 3):
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
        width=500,
        height=500,
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
            xaxis = dict(nticks=4, range=[0,400],),
            yaxis = dict(nticks=4, range=[0,300],),
            zaxis = dict(nticks=4, range=[0,24],),
        ),
    )
    
    

    fig.show()

def Track_3D_subplot(df_list,mac,df_wifipos,marker_size = 3,line_width = 3):
    track_list = []
    for i,df in enumerate(df_list):
        df_now = utils.GetDfNow(df,mac)
        x,y,z = GetOriginXYZ(df_now,df_wifipos)
        track_list.append([[x,y,z],f"epoch{i}"])
    if len(track_list) == 0:
        return
    track_list[0][1] = "origin"
    track_list[len(track_list)-1][1] = "final"
    col_num = 3
    row_num = float.__ceil__(len(track_list)/col_num)

    #get specs
    list_specs = []
    for i in range(row_num):
        l = []
        for j in range(col_num):
            l.append({'type':'scatter3d'})
        list_specs.append(l)

    #get name tuple
    name_list = []
    for i in range(len(track_list)):
        name_list.append(track_list[i][1])
    name_tuple = tuple(name_list)

    fig = make_subplots(
        rows=row_num, cols=col_num,
        specs=list_specs,
        subplot_titles=name_tuple
    )

    for i in range(len(track_list)):
        t = track_list[i]
        row_loc = int(i/col_num)+1
        col_loc = i%col_num+1
        fig.add_trace(
            go.Scatter3d(
                x=t[0][0], 
                y=t[0][1], 
                z=t[0][2],
                
                marker=dict(
                    color=z,
                    colorscale='Viridis',
                    size=marker_size,
                ),
                line=dict(
                    color='rgba(50,50,50,0.6)',
                    width=line_width,
                ),
            ),
            row=row_loc, col=col_loc,
        )
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
        ),
        row=row_loc, col=col_loc,)
    fig.update_layout(
        title_text='Track Restore Result',
        height=350*row_num,
        width=1000,
        showlegend = False
    )

    fig.update_scenes(xaxis = dict(nticks=4,range=[0,400]),
                      yaxis = dict(nticks=4,range=[0,300]),
                      zaxis = dict(nticks=4,range=[0,24]),
                        aspectratio=dict(x=1, y=0.75, z=0.75),
                        
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
                            ),
                        ),
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="hour" ,
                        )

    fig.show()

def Track3D_Origin(df_now,df_wifiPos):
    x,y,z = GetOriginXYZ(df_now,df_wifiPos)
    Track_3D(x,y,z)

def GetOriginXYZ(df_now,df_wifiPos):
    z = []
    x = []
    y = []
    for index,row in df_now.iterrows():
        z.append(row.t.hour+(row.t.minute/60))
        x.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].X)
        y.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].Y)
    return x,y,z

def Track3D_Virtual(df_now,df_wifiPos):
    x,y,z = GetVirtualXYZ(df_now,df_wifiPos)
    Track_3D(x,y,z)

def GetVirtualXYZ(df_now,df_wifiPos):
    z = []
    x = []
    y = []
    for index,row in df_now.iterrows():
        z.append(row.t.hour+(row.t.minute/60))
        x.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].restored_x)
        y.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].restored_y)
    return x,y,z

def GetPath(start,end,df_path):
        for i,row in df_path.iterrows():
            starts,destis = row.path.split('->')
            starts = starts.split(':')
            destis = destis.split(':')
            if str(start) in starts and str(end) in destis:
                return [x for x in row[1:] if str(x) != 'nan']
        return None

def Track3D_Restored(df_now,df_wifipos,df_path):
    def _append_pos(df_wifipos,tracker,time):
        z.append(time)
        x.append(df_wifipos[df_wifipos.wifi == tracker].iloc[0].restored_x)
        y.append(df_wifipos[df_wifipos.wifi == tracker].iloc[0].restored_y)

    def _append_path(df_path,start,end,start_time,end_time):
        path_points = GetPath(start,end,df_path)
        
        if len(path_points) == 0:
            return
        length = len(path_points)
        for i,point in enumerate(path_points):
            z.append(start_time + (i+1)*(end_time-start_time)/(length+1))
            xx,yy = point.split(':')
            x.append(float(xx))
            y.append(float(yy))

    x = []
    y = []
    z = []
    wifi_last = -1
    for index,row in df_now.iterrows():
        if index == 0:
            wifi_last = row.a
            continue  
        if row.a == wifi_last:
            _append_pos(df_wifipos,row.a,row.t.hour+(row.t.minute/60))
        else:
            row_last = df_now.iloc[index-1]
            time_start = row_last.t.hour+(row_last.t.minute/60)
            time_end = row.t.hour+(row.t.minute/60)
            _append_path(df_path,row_last.a,row.a,time_start,time_end)
            wifi_last = row.a
    Track_3D(x,y,z,marker_size=2,line_width=6)

def _addTrackCount(track_list,track_count,track,add_num = 1):
        for i,t in enumerate(track_list):
            if t == track or t == (track[1],track[0]):
                track_count[i] += add_num
                return True
        return False

def _getPathAndStay(df_now,df_wifipos,df_path,pass_path,pass_count,stay_pos,stay_count):
    wifi_last = -1
    for index,row in df_now.iterrows():
        if index == 0:
            wifi_last = row.a
            continue
        #stay at same place ?
        if row.a == wifi_last:
            last_time = (row.t - df_now.iloc[index-1].t).total_seconds()
            loc = utils.GetRestoredLocation(df_wifipos,row.a)
            if _addTrackCount(stay_pos,stay_count,loc,last_time) == False:
                stay_pos.append(loc)
                stay_count.append(last_time)
        #place changed
        else:
            paths = GetPath(wifi_last,row.a,df_path)
            last_loc = 0
            for i in range(len(paths)):
                loc_str = paths[i].split(':')
                loc = [float(loc_str[0]),float(loc_str[1])]
                if i == 0:
                    last_loc = loc
                    continue
                path = (last_loc,loc)
                if _addTrackCount(pass_path,pass_count,path) == False:
                    pass_path.append(path)
                    pass_count.append(1)
                last_loc = loc
            wifi_last = row.a

def Track2D_Restored(df,df_wifipos,df_path,show_freq = True):
    mac_list = df.m.unique()

    pass_path= [] # [tuple1([x1,y1],[x2,y2]),tuple2([x1,y1],[x2,y2])...]"
    pass_count = []
    stay_pos = [] # [[x1,y1],[x2,y2]...]
    stay_count = []
    
    for mac in tqdm(mac_list):
        df_now = utils.GetDfNow(df,mac)
        _getPathAndStay(df_now,df_wifipos,df_path,pass_path,pass_count,stay_pos,stay_count)
    
    
    if show_freq:
        Track_2D(pass_path,pass_count,stay_pos,stay_count)

def Track_2D(pass_path, pass_count=None, stay_pos=None, stay_count=None):

    fig = go.Figure()
    if pass_count != None:
        # normalize counts
        if max(pass_count)>50:
            pass_count = utils.Normalize_arr(pass_count)
            min_pass_count = min(pass_count) if min(pass_count)>0.1 else 0.1
            pass_count = (1/min_pass_count)*pass_count
        stay_count = utils.Normalize_arr(stay_count)*20

        #add move trace
        for i, path in enumerate(pass_path):
            xx = [path[0][0], path[1][0]]
            yy = [path[0][1], path[1][1]]
            fig.add_trace(go.Scatter(
                x=xx, y=yy,
                line=dict(color='firebrick', width=pass_count[i]),
                showlegend=False,
                mode='lines',
            ))
        #add stay trace
        xx = []
        yy = []
        for i,pos in enumerate(stay_pos):
            xx.append(pos[0])
            yy.append(pos[1])
        fig.add_trace(
            go.Scatter(x = xx,y = yy,
                        marker_size = stay_count,
                       mode="markers",
                       marker_color = "firebrick",
                       showlegend=False)
        )

    fig.update_layout(
        xaxis=dict(range=[0, 400]),  # Set x-axis range
        yaxis=dict(range=[0, 300]),  # Set y-axis range
        width=400, height=300,
        margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=4
        ),
        
    )

    fig.add_layout_image(
            dict(
                source=background_img,
                xref="x", yref="y",
                x=0, y=0,  #position of the upper left corner of the image in subplot 1,1
                sizex= 400,sizey= 300, #sizex, sizey are set by trial and error
                xanchor="left",
                yanchor="bottom",
                sizing="stretch",
                layer="below",
                opacity=0.3)
    )
    fig.show()

