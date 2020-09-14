# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 10:57:52 2020

@author: rbisa
"""
from flask import Flask

import dash
import dash_html_components as html
import dash_core_components as dcc

app = Flask(__name__)

#first open the file with the model; unpickle it


#route 1: "Welcome to XXX PP POWER PREDICTION"


#route 2: "Please upload the PP maintenance schedule"


#route 3: solar PP model
#pulls in relevant sun weather, used to determine power output for the next 7 days using model_solar.py
#includes the maintenance schedule

#route 4: wind PP model
#pulls in relevant wind weather, used to determine power output for the next 7 days using model_wind.py
#includes the maintenance schedule

#route 5: DASH layout
#graphs showing 7-day expected power output; line graph could work per PP; solar + wind 
#separately

if __name__ == '__main__':
    app.debug = True