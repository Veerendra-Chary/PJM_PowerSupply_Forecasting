# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import streamlit as st
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px



model1 = pickle.load(open(r'C:\Users\cchar\Power supply Forecasting Project/Forecast_random_forest.pkl',"rb"))
daily_data_last_7=pd.read_csv(r'C:\Users\cchar\Power supply Forecasting Project/daily_data_last_7.csv', header=None)
data=pd.read_csv(r"C:\Users\cchar\Power supply Forecasting Project/future_df.csv",header=0, index_col=0, parse_dates=True)



st.title("Forecast power consumption data")
st.sidebar.subheader("Select the number of days to Forecast from 2018-Aug-4")
days = st.sidebar.number_input('Days',min_value = 1,step = 1)

z=daily_data_last_7
z=np.array(z[0].tail(7))
for i in range(0,days):
    r=z[-7:]
    r=np.array([r])
    ranf_f=model1.predict(r)
    z=np.append(z,ranf_f)
    i=+1
future_pred=z[-days:]

    
    
future = pd.date_range(start='8/4/2018',periods=days,tz=None,freq = 'D')
future_df = pd.DataFrame(index=future)
future_df['Power Consumption'] = future_pred.tolist()

st.sidebar.write(f"Power consumption for {days}th day")
st.sidebar.write(future_df[-1:])

st.write(f"Power consumptionForecasted till {days}" )
st.write(future_df)

st.subheader('Actual and Forecast plot')
fig=plt.figure(figsize=(12,6))
px.line(data[-365:].index,data[-365:].values)
plt.plot(future_df.index,future_df.values)
plt.xlabel('Year',fontsize=14)
plt.ylabel('Power consumption',fontsize=14)
st.pyplot(fig)