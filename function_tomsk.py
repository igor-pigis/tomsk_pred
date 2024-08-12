import numpy as np

import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, plot, iplot

with open('id_dict.pkl', 'rb') as f:
    id_dict = pickle.load(f)
    
def smape(a,f):
    return (1/len(a))*np.sum(2*np.abs(f-a)/(np.abs(a) + np.abs(f)))

    
def show_forecast(cmp_df, num_predictions, num_values, id):
    # верхняя граница доверительного интервала прогноза
    '''upper_bound = go.Scatter(
        name='Upper Bound',
        x=cmp_df.tail(num_predictions).index,
        y=cmp_df.tail(num_predictions).yhat_upper,
        mode='lines',
        #marker=dict(color="444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')'''

    # прогноз
    forecast = go.Scatter(
        name='Prediction',
        x=cmp_df.tail(num_predictions).index,
        y=cmp_df.tail(num_predictions).yhat,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    )

    # нижняя граница доверительного интервала
    '''lower_bound = go.Scatter(
        name='Lower Bound',
        x=cmp_df.tail(num_predictions).index,
        y=cmp_df.tail(num_predictions).yhat_lower,
        #marker=dict(color="444"),
        line=dict(width=0),
        mode='lines')'''

    # фактические значения
    fact = go.Scatter(
        name='Fact',
        x=cmp_df.tail(num_values).index,
        y=cmp_df.tail(num_values).y,
        #marker=dict(color="red"),
        mode='lines',
    )
    
    # последовательность рядов в данном случае важна из-за применения заливки
    #data = [lower_bound, upper_bound, forecast, fact]
    data = [forecast, fact]

    layout = go.Layout(
        yaxis=dict(title='Q'),
        title=f'Q для модели lr на {2*pred_len} h. Объект: {id_dict[id]}',
        showlegend = True,
        yaxis_range = [0,cmp_df.y.max()*1.1])

    #layout.update_layout(yaxis_range = [-4,4])
    #buf = io.BytesIO()
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)
    
    #plotly_file = f"lr_pred_{id}.html"
    #fig.write_html(plotly_file)
    #mlflow.log_artifact(plotly_file, artifact_path=f"lr_pred_{id}.html")
    #os.remove(f'lr_pred_{id}.html')

