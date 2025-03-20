# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:05:53 2025

@author: anna_
"""
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Load Data. The file path must be changed to the path where the folder is located on your computer
df_merged = pd.read_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\Merged_Data.csv')

df_2019 = pd.read_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\testData_2019_Central.csv')
df_2019["Date"] = pd.to_datetime(df_2019["Date"], format="%Y-%m-%d %H:%M:%S")
df_holidays = pd.read_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\holiday_17_18_19.csv', sep=",") # got the holiday data from the classroom 2021
df_holidays["Date_only"] = pd.to_datetime(df_holidays["Date"], format="%d.%m.%Y")


# To be sure that the datetime is right
df_2019['Date_only'] = pd.to_datetime(df_2019['Date']).dt.date
df_holidays['Date_only'] = pd.to_datetime(df_holidays['Date_only'], format="%d.%m.%Y").dt.date

#Rename columns
df_2019.rename(columns={'Date': 'datetime'}, inplace=True)
df_2019['datetime'] = pd.to_datetime(df_2019['datetime'])
df_2019.rename(columns={'Central (kWh)': 'Power_kW'}, inplace=True)
df_2019.drop(columns=['rain_day'], inplace=True)
df_2019['datetime'] = pd.to_datetime(df_2019['datetime'])
df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])

# Merging holidays and 2019 Data
df_2019 = pd.merge(df_2019, df_holidays[['Date_only']], on='Date_only', how='left')
df_2019['Holiday'] = df_2019['Date_only'].isin(df_holidays['Date_only']).astype(int)
df_2019.drop(columns=['Date_only'], inplace=True)
df_2019['weekday'] = df_2019['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
df_2019['hour'] = df_2019['datetime'].dt.hour

#Calculate daily average temperature and calculate Heating and Cooling Days
daily_avg_temp = df_2019.groupby(df_2019["datetime"].dt.date)["temp_C"].mean()
heating_days_flag = (daily_avg_temp < 16).astype(int)
cooling_days_flag = (daily_avg_temp > 21).astype(int)
df_2019["Heating_Day"] = df_2019["datetime"].dt.date.map(heating_days_flag)
df_2019["Cooling_Day"] = df_2019["datetime"].dt.date.map(cooling_days_flag)

#Feature calculations:
#Lag1 calculation
df_2019['temp_C_lag1_x'] = df_2019['temp_C'].shift(1)
df_2019['HR_lag1'] = df_2019['HR'].shift(1)
df_2019['windSpeed_m/s_lag1'] = df_2019['windSpeed_m/s'].shift(1)
df_2019['Power_kW_lag1'] = df_2019['Power_kW'].shift(1)

#Sin and Cos calculation
df_2019['sin_hour'] = np.sin(2 * np.pi * df_2019['hour'] / 24)
df_2019['cos_hour'] = np.cos(2 * np.pi * df_2019['hour'] / 24)

df_2019['temp_HR_interaction'] = df_2019['temp_C'] * df_2019['HR']

df_2019.ffill(inplace=True)

# convert dataframe and calculate the NaN Data in the first row for the lag1 numbers
df_2019['datetime'] = pd.to_datetime(df_2019['datetime'])
df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])

# Get the Power_kW value of December 31, 2018, 23:00 from df_merged
df_2018_12_31_23 = df_merged[df_merged['datetime'].dt.date == pd.to_datetime('2018-12-31').date()]
df_2018_12_31_23 = df_2018_12_31_23[df_2018_12_31_23['datetime'].dt.hour == 23]
power_kW_31_12_23 = df_2018_12_31_23['Power_kW'].iloc[0]
temp_kw_31_12_23 = df_2018_12_31_23['temp_C'].iloc[0]
HR_kW_31_12_23 = df_2018_12_31_23['HR'].iloc[0]
windspeed_kw_31_12_23 = df_2018_12_31_23['windSpeed_m/s'].iloc[0]

# Set this value for Power_kW_lag1_x and other columns in df_2019 for January 01, 2019, 00:00 h
df_2019.loc[(df_2019['datetime'].dt.date == pd.to_datetime('2019-01-01').date()) & 
            (df_2019['datetime'].dt.hour == 0), 'Power_kW_lag1'] = power_kW_31_12_23

df_2019.loc[(df_2019['datetime'].dt.date == pd.to_datetime('2019-01-01').date()) & 
            (df_2019['datetime'].dt.hour == 0), 'temp_C_lag1_x'] = temp_kw_31_12_23

df_2019.loc[(df_2019['datetime'].dt.date == pd.to_datetime('2019-01-01').date()) & 
            (df_2019['datetime'].dt.hour == 0), 'HR_lag1'] = HR_kW_31_12_23

df_2019.loc[(df_2019['datetime'].dt.date == pd.to_datetime('2019-01-01').date()) & 
            (df_2019['datetime'].dt.hour == 0), 'windSpeed_m/s_lag1'] = windspeed_kw_31_12_23


df_2019.rename(columns={'temp_C_lag1_x': 'temp_C_lag1'}, inplace=True)

# Checks if there are any NaN Values
print(df_2019.isnull().sum())

# This line says where the pickle file is saved
project_folder = "Project 1 Energy Services IST1115467"

# find the complete path for the pickle file
rf_model_path = os.path.join(project_folder, "random_forest_model.pkl")

# Check whether the file exists before you load it
if os.path.exists(rf_model_path):
    with open(rf_model_path, "rb") as f:
        loaded_rf_model = pickle.load(f)
    print("Random Forest Modell found and loaded.")
else:
    print(f" Error: Can't find the file with the {rf_model_path}'!")

print("Random Forest Modell successfully loaded.")
important_features = [ 'Power_kW_lag1', 'hour', 'cos_hour', 'solarRad_W/m2', 'temp_C']


# Match the feature names used during training
X_new = df_2019[loaded_rf_model.feature_names_in_]
df_2019['predicted_Power_kW'] = loaded_rf_model.predict(X_new)
y_true = df_2019['Power_kW']
y_pred = df_2019['predicted_Power_kW']

#Plot
plt.figure(figsize=(12, 6))
plt.plot(df_2019['datetime'], df_2019['Power_kW'], label="Actual Power (kW)", color='blue', alpha=0.7)
plt.plot(df_2019['datetime'], df_2019['predicted_Power_kW'], label="Predicted Power (kW)", color='red', linestyle='dashed')
plt.xlabel('Day and Time')
plt.ylabel('Power (kW)')
plt.xlim(df_2019["datetime"].min(), df_2019["datetime"].max())
plt.title('Actual vs Predicted Power Consumption 2019')
plt.legend()
plt.show()

# Scatter Plot of Actual vs Predicted Consumption
plt.figure(figsize=(8, 8))
sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel("Actual Consumption (kW)")
plt.ylabel("Predicted Consumption (kW)")
plt.title("Actual vs. Predicted Power Consumption with random forest regression")
plt.grid()
plt.show()

#Evaluation
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
r2 = r2_score(y_true, y_pred)
print(f"R²-Score: {r2}")

df_2019.to_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\Merged_data2019.csv', index=False) 
