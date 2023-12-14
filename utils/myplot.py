import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import os
img_path = os.getcwd()+'/wifi_track_data/dacang/imgs/roads.png'
img = Image.open(img_path)
background_img = img
buttom_img = Image.fromarray(np.array(img.transpose(Image.FLIP_TOP_BOTTOM))).convert('P', palette='WEB', dither=None)

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
                            line=dict(width=1)))
    
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
        width=500,
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
    
    if box_title == "":
        fig.update_layout(title_text="Box Plot")
    else :
        fig.update_layout(title_text=box_title)
    
    fig.show()


def Track_3D(x,y,z,x_name = "",y_name = "",z_name = ""):
    dum_img = Image.fromarray(np.ones((3,3,3), dtype='uint8')).convert('P', palette='WEB')
    idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
    x=x, 
    y=y, 
    z=z,
    
    marker=dict(
        color=z,
        colorscale='Viridis',
        size=3,
    ),
    line=dict(
        color='rgba(50,50,50,0.6)',
        width=3,
        
    )
    ))

    im_x = np.linspace(0, 400, 400)
    im_y = np.linspace(0, 300, 300)
    im_z = np.zeros((300,400))

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
                zaxis_title="Z" ,
            aspectmode = 'manual',
            aspectratio=dict(x=1, y=0.75, z=0.75),
            xaxis = dict(nticks=4, range=[0,400],),
            yaxis = dict(nticks=4, range=[0,300],),
            zaxis = dict(nticks=4, range=[0,24],),
        ),
    )
    
    

    fig.show()

def Track3D_Origin(df_now,df_wifiPos):
    z = []
    x = []
    y = []
    for index,row in df_now.iterrows():
        z.append(row.t.hour+(row.t.minute/60))
        x.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].X)
        y.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].Y)
    Track_3D(x,y,z)

def Track3D_Virtual(df_now,df_wifiPos):
    z = []
    x = []
    y = []
    for index,row in df_now.iterrows():
        z.append(row.t.hour+(row.t.minute/60))
        x.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].restored_x)
        y.append(df_wifiPos[df_wifiPos.wifi == row.a].iloc[0].restored_y)
    Track_3D(x,y,z)