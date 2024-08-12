#streamlit run uber_pickups.py
import pickle
import pandas as pd
import numpy as np

import streamlit as st
import importlib
import datetime
from datetime import timedelta, datetime
from sklearn.linear_model import HuberRegressor
import os
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, plot, iplot

import matplotlib.pyplot as plt
###################################################################
# импорт функций сглаживания Q
import smoothfunction 
importlib.reload(smoothfunction)

import create_features 
importlib.reload(create_features)

import train_and_pred
importlib.reload(train_and_pred)

#import function_tomsk
#importlib.reload(function_tomsk)

# загружаем словарь номер объекта : функция сглаживания 
with open('smooth_dict.pkl', 'rb') as f:
    smooth_dict = pickle.load(f)

# загружаем словарь номер объекта : название объекта (формируется словарь при выполнении скрипта data_preparation.py)
with open('id_dict.pkl', 'rb') as f:
    id_dict = pickle.load(f)
####################################################################
input_date = []
input_hour = None
smooth_select = False
pred_len = None
pred_start_date = None
trig = False
train_complite = None
smooth_complite = None
pred_complite = None
df_Q_sm = []
trig = False
start_button = False
#######################################################################
st.title('Предсказание газопотребления')
step = st.empty()    
# cчитываем данные по потреблению, температуре и праздникам 
df_Q = pd.read_csv('df_Q_w.csv', index_col = 'Date', parse_dates = True)
   
df_T = pd.read_csv('df_T_w.csv', index_col = 'Date', parse_dates = True)

df_holidays = pd.read_csv('df_holidays.csv')
df_holidays['Date'] = pd.to_datetime(df_holidays['Date'])
#########################################################################
id_select = []    
id_select = st.selectbox(
    "Выберите ГРС для прогноза",
    (id_dict.values()),
    index=None,
    placeholder="ГРС...",
)

if id_select != None:
    id = list(id_dict.keys())[list(id_dict.values()).index(id_select)]
    st.write(f"Был выбран: {id_select} |id = {id}")
#######################################################################
if id_select != None:
    models_dir = f'saved_models_{id}' # место хранения моделей
    if not os.path.exists(models_dir): # cоздается директорию для сохранения моделей, если она еще не существует
        os.makedirs(models_dir)
##########################################################################
if (id_select != None):
    st.write('Ряд без сглаживания')
    st.line_chart(df_Q[df_Q.index > '2022-01-01' ], y = id, x_label = 'Время', y_label = 'Газопотребление, ???')
    st.write('Сглаженный ряд')
    st.line_chart(smooth_dict[id](df_Q[df_Q.index > '2022-01-01' ][id]), x_label = 'Время', y_label = 'Газопотребление, ???')
###########################################################################
#Применяем функцию сглаживания из smoothfunction
if (id_select != None):
    smooth_select = st.selectbox(
        "Выберете сглаживать ли данные",
        ['Да','Нет'],
        index=None,
        placeholder="...",
    )
    if smooth_select:
        st.write("Данные будут сглажены")
    elif smooth_select == False:
        st.write("Данные не будут сглажены")

if (id_select != None) & (smooth_select != None):
    df_Q_sm = df_Q.copy()
    if smooth_select == True:
        #df_Q[id] = smoothfunction.smooth_anomalies(df_Q[id])
        df_Q_sm[id] = smooth_dict[id](df_Q[df_Q.index > '2022-01-01' ][id])
        smooth_complite = True
    else: 
        df_Q_sm[id] = df_Q[df_Q.index > '2022-01-01' ][id]
        smooth_complite = True
########################################################################
if (id_select != None) & (smooth_complite != None):
    min_date = pd.to_datetime('2024-01-01', format = '%Y-%m-%d')
    max_date = df_Q.index.max() - pd.Timedelta(days = 30)
    input_date = st.date_input("Выберите дату начала прогноза", value = None, min_value = min_date, max_value = max_date )
    st.write("Прогнозирование с ", str(input_date))
    

    input_hour = []    
    input_hour = st.selectbox(
        "Выберите час начала прогноза",
        (['01:00','03:00','05:00','07:00','09:00','11:00','13:00','15:00','17:00','19:00','21:00','23:00']),
        index=None,
        placeholder="01:00",
        )

    input_hour = input_hour + ':00'
    #input_hour = st.time_input("Выберите час начала прогноза", value =  None, step = 3600) 
    st.write(f"Прогнозирование с {str(input_hour)[-8:-6]} часов")
    
    #st.write(input_date + timedelta(hours = int(str(input_hour)[-8:-6])))
    #st.write(int(str(input_hour)[-8:-6]))
if (id_select != None) & (input_date != None) & (input_hour != None):
    pred_start_date = datetime.strptime(str(input_date) + ' ' + str(input_hour), '%Y-%m-%d %H:%M:%S')
    #st.write(str(pred_start_date))
###########################################################################
if (input_hour != None) & (smooth_complite != None):
    pred_len = st.select_slider(
        "Выберете горизонт прогнозирования в сутках",
        options=range(1,11),
        help = 'Длительность в сутках на которую будет считаться прогноз начиная с введенной выше даты. Например, если введенная дата `2024-01-01 01:00:00` и выбрать `горизонт = 5 суток`, то мы получим прогноз на период `2024-01-01 01:00:00` - `2024-01-06 01:00:00`',
        value = 10
    )
    if pred_len != None:
        st.write(f"Прогноз строится на {pred_len} суток начиная с {pred_start_date}")       
############################################################################
if (input_hour != None) & (smooth_complite != None):
     start_button = st.toggle("Выполнить расчёт")
     #if start_button:
        #st.write(smooth_select)
######################################################################
#if (input_hour != None) & (smooth_complite != None) & (smooth_select != None):
#   # st.button("Отмена расчёта", type="primary")
#    st.button("Провести расчёт", on_click  )
#    if st.button("Провести расчёт"):
#        start_button = True
#        st.button("Отмена расчёта", disabled = not start_button)
#    else:
#        start_button = False
############################################################################
if (pred_len != None) & (start_button != False) & (pred_start_date != None):
    #train_bar_text = "Идет обучение моделей. Пожалуйста подождите."
    #train_bar = st.progress(0, text=train_bar_text)

    df_features_without_lag = create_features.create_features_without_lag(df_Q_sm[id], df_T[id], df_holidays, id, trig = False)
    df_features_without_lag = df_features_without_lag[df_features_without_lag.index > '2022-01-01' ]
   
    cmp_df_id = pd.DataFrame()
    mape = []
    mae = []
    smape = []
    index_pred = []
    predictions = []
    pred_bar_text = "Считается прогноз"
    pred_bar = st.progress(0, text=pred_bar_text)
    for i in range(1, pred_len*12+1):
        prediction_, date_target_ = train_and_pred.train_and_pred(df_features_without_lag, id = id, i = i, date_start = pred_start_date, models_dir = models_dir)
        predictions.append(float(prediction_))
        index_pred.append(date_target_)
    
        time.sleep(0.01)
        pred_bar.progress(i/(pred_len*12), text=pred_bar_text)
        #time.sleep(1)
    pred_bar.empty()
    pred_complite = True
    st.write('Прогноз посчитан')

    cmp_df_i = pd.DataFrame(index = index_pred)
    cmp_df_i.loc[index_pred, 'Фактическое потребление'] = df_Q_sm.loc[index_pred][id]
    cmp_df_i.loc[index_pred, 'Прогноз'] = predictions
    
    # расчет и вывод метрик
    metrics_i_ = {}

    mae_i = mean_absolute_error(df_Q_sm.loc[index_pred][id], predictions)
    mape_i = mean_absolute_percentage_error(df_Q_sm.loc[index_pred][id], predictions)
    #smape_i = (1/len(df_Q_sm.loc[index_pred][id]))*np.sum(2*np.abs(predictions-df_Q_sm.loc[index_pred][id])/(np.abs(df_Q_sm.loc[index_pred][id]) + predictions))
    mape.append(mape_i)
    mae.append(mae_i)
    #smape.append(smape_i)
    
    cmp_df_id = pd.concat([cmp_df_id, cmp_df_i])
    
    metrics_lr = {}
    
    metrics_lr['MAE'] = np.mean(mae)
    metrics_lr['MAPE, %'] = np.mean(mape)*100
    st.write('Ошибки прогноза:', pd.DataFrame(metrics_lr, index = ['']))
###################################################################
if (pred_complite == True):
    #st.line_chart(cmp_df_id, x_label = 'Время', y_label = 'Прогноз газопотребления, ???')
    
    #fig = plt.figure() 
    #plt.plot(cmp_df_id) 
    #st.pyplot(fig)
    
    #st.write(cmp_df_id)

    #create your figure and get the figure object returned
    fig, ax = plt.subplots(figsize = (10,5))
    ax.plot(np.array([x for x in range(pred_len*12)]),cmp_df_id)

    ax.set_xlabel('Время от даты начала прогноза, ч')
    ax.set_ylabel('Газопотребление, ???') 
    ax.legend(['Фактическое потребление','Прогноз'], loc = 'lower right')
    #ax.set_xlim(1,pred_len*12+1)
    #ax.set_xticks(range(1,pred_len*12+1))
    ax.set_ylim(0, cmp_df_id.iloc[:,1].max()*1.1)
    st.pyplot(fig) # instead of plt.show()
