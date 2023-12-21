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