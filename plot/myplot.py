import plotly.express as px
import plotly.io as pio

def Mac_Dur_Count_Scatter(df):
    fig = px.scatter(df,x="Dur",y="A_count",size="Count",color="oriMac",hover_name="M")
    fig.update_xaxes(title='24h内持续时间')
    fig.update_yaxes(title="到达的探针数量")
    fig.show()