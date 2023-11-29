import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

colors = ['rgb(67,67,67)', 'rgb(115,115,115)']
line_size = [2,2,2,2]

def Mac_Dur_Count_Bubble_Scatter(df):
    fig = px.scatter(df,x="Dur",y="A_count",size="Count",color="oriMac",hover_name="M")
    fig.update_xaxes(title='24h内持续时间')
    fig.update_yaxes(title="到达的探针数量")
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

def Date_Mac_3D_scatter(df,x_name,y_name,z_name,species_name):
    fig = px.scatter_3d(df, x=x_name, y=y_name, z=z_name,
              color=species_name, 
               opacity=0.7)

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
