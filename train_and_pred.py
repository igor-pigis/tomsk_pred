import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import time
import datetime

import pickle
import joblib
import os
import holidays
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import HuberRegressor
import create_features
import streamlit as st

def train_and_pred(df_features_without_lag, id, i, date_start, models_dir):
    model = []
    trig = False
    if trig == False:
        non_scaling_features = ['holiday',
                            'month_1', 'month_2', 'month_3',
           'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
           'month_10', 'month_11', 
                            'day_of_week_1', 'day_of_week_2',
           'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
           'Hour_3', 'Hour_5', 'Hour_7', 'Hour_9', 'Hour_11', 'Hour_13',
           'Hour_15', 'Hour_17', 'Hour_19', 'Hour_21', 'Hour_23']
    if trig == True:
        non_scaling_features = ['holiday','month_sin','month_cos','dow_sin','dow_cos','hour_sin','hour_cos']
    
    date_target = date_start - pd.Timedelta(hours=2)

    # экстракция признаков из временного ряда
    df_features = create_features.create_features_lag(df_features_without_lag, i) 
        
    # определение тренировочного набора
    train_size = len(df_features.loc[df_features.index <= date_target])
    train = df_features.iloc[:train_size]

    y_train = train['Q']  
    X_train = train.drop('Q', axis = 1)
    
    # масштабирование данных
    X_scaler = RobustScaler()
    y_scaler = RobustScaler()

    y_train_norm = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    
    X_train_norm = X_scaler.fit_transform(X_train.drop(non_scaling_features, axis = 1))
    X_train_norm = np.c_[X_train_norm, X_train[non_scaling_features]]
    
    model = HuberRegressor()
    # обучение модели
    model.fit(X_train_norm, y_train_norm)

    # строка, которая на i*2 часов больше чем date_start
    date_target = date_target + pd.Timedelta(hours=i*2)
        
    # определение тестового набора
    test_ = df_features.iloc[df_features.index == date_target]
    y_test_ = test_['Q']
    X_test_ = test_.drop('Q', axis = 1)
    #st.write(y_test_)
    
    X_test_norm_ = X_scaler.transform(X_test_.drop(non_scaling_features, axis=1))
    X_test_norm_ = np.c_[X_test_norm_, X_test_[non_scaling_features]]
    
    # предсказание
    y_pred_scaled_ = model.predict(X_test_norm_)
    y_pred_ = y_scaler.inverse_transform(y_pred_scaled_.reshape(-1, 1)).ravel()
    #st.write(y_pred_)
    return y_pred_, date_target