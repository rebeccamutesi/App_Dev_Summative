# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:02:18 2020

@author: rbisa
"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#loading the data for analysis
wind_data = pd.read_csv('wind_generation_data.csv')

#cleaning the data
x_wind = wind_data.isnull().sum(axis=0) 
print(x_wind)

#adding date column to wind dataframe
wind_data['Date'] = pd.date_range(start='1/1/2016', periods=len(wind_data), freq='D')

#checking datatypes
y_wind = wind_data.info()

#ML modelling: XGBoost 
dataset = wind_data.drop(['Power Output', 'Date'], axis=1)
X_wind = dataset.values
y_wind = wind_data['Power Output'].values

#data splitting

X_train, X_test, y_train, y_test = train_test_split(X_wind, y_wind, test_size=0.3, random_state=42)

#data transformation (scaling)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#creation of regressor model

xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

#fitting model
xgb_model.fit(X_train, y_train) # fit model

#predicting
y_predicted_w = xgb_model.predict(X_test)

#accuracy determination of random forest regression
rmse = np.sqrt(mean_squared_error(y_test, y_predicted_w))

pickle.dump(xgb_model, open('model_w.pkl','wb'))