# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:02:32 2020

@author: rbisa
"""
#import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#loading the data for analysis
solar_data = pd.read_csv('solar_generation_data.csv')

#cleaning the data
x = solar_data.isnull().sum(axis=0) 
print(x)

#filling the NaN values with zeros in the rainfall column
solar_data['Rainfall in mm'] = solar_data['Rainfall in mm'].fillna(0)

#removing the white space in the "month" column
solar_data.rename(columns=lambda x:x.replace(' ', '') if ' ' in x else x, inplace=True)

#creating datetime for this dataset
#adding a column with 2019 as default year
solar_data_2 = solar_data.assign(Year = '2019')

#converting month to integer
c = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6, 'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
solar_data_2.Month = solar_data_2.Month.str.upper().map(c)

solar_data_2['Date'] = pd.to_datetime(solar_data_2[['Year', 'Month', 'Day']])

#checking datatypes
y = solar_data_2.info()

#changing data to suitable types for ML model
solar_data_2['TempHi'] = solar_data_2['TempHi'].replace('\u00b0','', regex=True)
solar_data_2['TempHi'] = pd.to_numeric(solar_data_2['TempHi'], downcast="float")

solar_data_2['TempLow'] = solar_data_2['TempLow'].replace('\u00b0','', regex=True)
solar_data_2['TempLow'] = pd.to_numeric(solar_data_2['TempLow'], downcast="float")

#rechecking datatypes
y = solar_data_2.info()

#ML section
#training and test sets

# Values of attributes

dataset = solar_data_2.drop(['Month', 'Day', 'Date', 'PowerGeneratedinMW','Rainfallinmm'], axis=1)
X = dataset.values
y = solar_data_2['PowerGeneratedinMW'].values

#data splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#data transformation (scaling)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#creation of regressor model

forest_model = RandomForestRegressor(n_jobs=-1)

#fitting model
forest_model.fit(X_train, y_train) # fit model

#predicting
y_predicted = forest_model.predict(X_test)

#accuracy determination of random forest regression
score = r2_score(y_test, y_predicted)

pickle.dump(forest_model, open('model.pkl','wb'))

#model = pickle.load(open('model.pkl','rb'))


