# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:59:35 2025

@author: anna_
"""

from scipy.spatial import KDTree
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import pickle

# Read files and safe them in a DataFrame. The file path must be changed to the path where the folder is located on your computer
df_2017 = pd.read_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\IST_Central_Pav_2017.csv', sep=",")
df_2018 = pd.read_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\IST_Central_Pav_2018.csv', sep=",")
df_2019 = pd.read_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\testData_2019_Central.csv')
df_meteo = pd.read_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\IST_meteo_data_2017_2018_2019.csv', sep=",")
df_holidays = pd.read_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\holiday_17_18_19.csv', sep=",") # got the holiday data from the classroom 2021

# Convert Datetime columns so every Dataframe is correctly declared
df_2017["Date_start"] = pd.to_datetime(df_2017["Date_start"], format="%d-%m-%Y %H:%M")
df_2018["Date_start"] = pd.to_datetime(df_2018["Date_start"], format="%d-%m-%Y %H:%M")
df_meteo["yyyy-mm-dd hh:mm:ss"] = pd.to_datetime(df_meteo["yyyy-mm-dd hh:mm:ss"], format="%Y-%m-%d %H:%M:%S")
df_meteo.rename(columns={"yyyy-mm-dd hh:mm:ss": "datetime"}, inplace=True)
df_holidays["Date"] = pd.to_datetime(df_holidays["Date"], format="%d.%m.%Y")

# Merge the energy Data from 2017 and 2018
df_energy = pd.concat([df_2017, df_2018], ignore_index=True)
df_energy = df_energy.sort_values(by="Date_start")

# Filter Meteo Data for 2017 and 2018
df_meteo = df_meteo[(df_meteo["datetime"] >= "2017-01-01") & (df_meteo["datetime"] < "2019-01-01")]


# Resample Meteo Data to Hourly Values because they are given in different time steps
df_meteo.set_index("datetime", inplace=True)
df_meteo_hourly = df_meteo.resample('H').agg({
    'temp_C': 'mean', 
    'HR': 'mean', 
    'windSpeed_m/s': 'mean', 
    'windGust_m/s': 'mean',
    'pres_mbar': 'mean',
    'solarRad_W/m2': 'mean',
    'rain_mm/h': 'sum' 
}).copy()

df_meteo_hourly.reset_index(inplace=True)

# Merge Energy Data with Hourly Meteo Data
df_energy.rename(columns={"Date_start": "datetime"}, inplace=True)
df_merged = pd.merge(df_energy, df_meteo_hourly, on="datetime", how="left")

# Check if there are NaN values
nan_count = df_merged.isna().sum()
print("Missing values in each column:")
print(nan_count[nan_count > 0])

# Fill Missing Meteo Data with Nearest Available Data (Nearest means with nearest energy consumption to the consumpteion where meteo data is missing)
missing_indices = df_merged[df_merged['temp_C'].isna()] 
completerows = df_merged.dropna()

if not completerows.empty:
    # Build KDTree using 'Consumption_kWh' from complete rows
    tree = KDTree(completerows[['Power_kW']].values)

    def fill_missing_values(row):
        if pd.isna(row['temp_C']):  # Only process rows with missing values
            _, idx = tree.query([[row['Power_kW']]])  # Find nearest neighbor
            idx = int(idx)  # Ensure it's an integer index

            # Verify idx is within bounds
            if 0 <= idx < len(completerows):
                nearest_row = completerows.iloc[idx]  # Get the nearest complete row
                return row.fillna(nearest_row)  # Fill missing values

        return row  # Return row unchanged if no missing values
    df_merged = df_merged.apply(fill_missing_values, axis=1)    

# Merge Holiday Data to the merged dataframe
df_merged['Date_only'] = df_merged['datetime'].dt.date
df_holidays['Date_only'] = df_holidays['Date'].dt.date
df_merged = pd.merge(df_merged, df_holidays[['Date_only']], on='Date_only', how='left')
df_merged['Holiday'] = df_merged['Date_only'].isin(df_holidays['Date_only']).astype(int)
df_merged.drop(columns=['Date_only'], inplace=True)

# Check for NaN values, did everything went right in the KDTree?
print("Total missing values after filling:", df_merged.isna().sum().sum())  

# Add Weekdays to the dataframe
df_merged["weekday"] = df_merged["datetime"].dt.dayofweek

# Extract the time
df_merged["hour"] = df_merged["datetime"].dt.hour

# Add if its day or night ( Day is between 6 am and 9pm) 
#I choosed this times because of the "Avergae energy consumption per hour over the course of the day" plot below
df_merged["Day"] = df_merged["hour"].apply(lambda x: 1 if 6 <= x <= 21 else 0)

#Add Heating and Cooling Days
# Calculate daily average temperature
daily_avg_temp = df_merged.groupby(df_merged["datetime"].dt.date)["temp_C"].mean()

# calculate Heating Days & Cooling Days how much is the difference?
heating_days_difference = (16 - daily_avg_temp).clip(lower=0)  # If below 16°C, otherwise 0
cooling_days_difference = (daily_avg_temp - 21).clip(lower=0)  # If above 21°C, otherwise 0

# Transfer values back into the hourly DataFrame
df_merged["Heating_Day_difference"] = df_merged["datetime"].dt.date.map(heating_days_difference)
df_merged["Cooling_Day_difference"] = df_merged["datetime"].dt.date.map(cooling_days_difference)

# Add new columns for “Heating_Day” and “Cooling_Day” (1 if it is one of the days mentioned, otherwise 0)
heating_days_flag = (daily_avg_temp < 16).astype(int)
cooling_days_flag = (daily_avg_temp > 21).astype(int)
df_merged["Heating_Day"] = df_merged["datetime"].dt.date.map(heating_days_flag)
df_merged["Cooling_Day"] = df_merged["datetime"].dt.date.map(cooling_days_flag)

# Review of the first 24 hours
print(df_merged[["datetime", "temp_C", "Heating_Day_difference", "Cooling_Day_difference"]].head(24))

# Feature Engineering
# Lag Features
df_merged['temp_C_lag1'] = df_merged['temp_C'].shift(1)
df_merged['HR_lag1'] = df_merged['HR'].shift(1)
df_merged['windSpeed_m/s_lag1'] = df_merged['windSpeed_m/s'].shift(1)
df_merged['Power_kW_lag1'] = df_merged['Power_kW'].shift(1)

# Fourier Features for Seasonality
df_merged['sin_hour'] = np.sin(2 * np.pi * df_merged['hour'] / 24)
df_merged['cos_hour'] = np.cos(2 * np.pi * df_merged['hour'] / 24)

# Interaction Features
df_merged['temp_HR_interaction'] = df_merged['temp_C'] * df_merged['HR']

# Time-based features
df_merged['hour'] = df_merged['datetime'].dt.hour
df_merged['month'] = df_merged['datetime'].dt.month

# Adding Holidays feature
df_merged['Date_only'] = df_merged['datetime'].dt.date
df_holidays['Date_only'] = df_holidays['Date'].dt.date
df_merged = pd.merge(df_merged, df_holidays[['Date_only']], on='Date_only', how='left')
df_merged['Holiday'] = df_merged['Date_only'].isin(df_holidays['Date_only']).astype(int)
df_merged.drop(columns=['Date_only'], inplace=True)

# Fill in the values with NaN values. The first row of the lag1 data is affected by this and the mean values of the entire column are assumed
df_merged.fillna(df_merged.mean(), inplace=True)

#Save Cleaned Data
df_merged.to_csv(r'C:\Dokumente\Uni Master\Lisboa\Energy Sources\merged_data.csv', index=False)

print("Data processing completed and saved!")

# Plotting
# Time series of energy consumption
plt.figure(figsize=(12, 6))
plt.plot(df_merged["datetime"], df_merged["Power_kW"], label="Consumption (kW)", color='b')
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("Power consumption over time")
plt.xlim(df_merged["datetime"].min(), df_merged["datetime"].max())
plt.xlim(df_merged["datetime"].min(), df_merged["datetime"].max())
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xlim(df_merged["datetime"].min(), df_merged["datetime"].max())
plt.legend()
plt.grid()
plt.show()
#little drop in August. It's probably because of the summer break

# Time series of energy consumption by Weekday
plt.figure(figsize=(10, 6))
sns.barplot(x=df_merged["weekday"], y=df_merged["Power_kW"], order=[
    "0", "1", "2", "3", "4", "5", "6"], palette="coolwarm")
plt.xlabel("Weekday")
plt.ylabel("Average Power (kW)")
plt.title("Average Energy Consumption by Weekday")
plt.xticks(rotation=45)
plt.show()
#you can see that on the weekend the energy consumption is a bit lower then during the week. During the week it's almost the same consumption every day

# Average consumption over a day
avg_hourly_consumption = df_merged.groupby('hour')['Power_kW'].mean()
plt.figure(figsize=(12, 6))
plt.plot(avg_hourly_consumption.index, avg_hourly_consumption.values, label= "Average Power (kW)", marker='o', color='b')
plt.xlabel("Hour of the day")
plt.ylabel("Average Energy consumption(kW)")
plt.title("Average energy consumption per hour over the course of the day")
plt.xlim(df_merged["hour"].min(), df_merged["hour"].max())
plt.xticks(np.arange(0, 24, step=1))
plt.grid(True)
plt.show()
#Consumption is rising after 6 till 10 am and then it sinks after 3pm till 21 pm

# Influence of Heating und Cooling Days
plt.figure(figsize=(12, 5))
heating_days_count = df_merged["Heating_Day"].sum()  /24
cooling_days_count = df_merged["Cooling_Day"].sum()  /24
sns.barplot(x=["Heating Days", "Cooling Days"], y=[heating_days_count, cooling_days_count])
plt.ylabel("Amount of days")
plt.title("Amount of heating and cooling days")
plt.show()
#We have more heating days in those 2 years than cooling days

# Energy consumption per weekday
plt.figure(figsize=(12, 5))
sns.boxplot(x=df_merged["weekday"], y=df_merged["Power_kW"])
plt.xlabel("Weekday")
plt.ylabel("Consumption (kW)")
plt.title("Consumption per weekday")
plt.show()
# every weekday is almost the same, just on the weekend the consumption is lower and there are more outliers

# Train Regression Model
df_train = df_merged[df_merged['datetime'].dt.year.isin([ 2017,2018])]
df_test = df_merged[df_merged['datetime'].dt.year == 2018]

features = [
    'temp_C', 'HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar', 'solarRad_W/m2', 'rain_mm/h', 
    'Holiday', 'hour', 'weekday', 'temp_HR_interaction', 
    'temp_C_lag1', 'HR_lag1', 'windSpeed_m/s_lag1', 'Power_kW_lag1', 
    'sin_hour', 'cos_hour', 'Heating_Day', 'Cooling_Day'
]

X_train, y_train = df_train[features], df_train['Power_kW']
X_test, y_test = df_test[features], df_test['Power_kW']
target = "Power_kW"

# Ensure that both data records have the same columns
missing_cols = set(df_train.columns) - set(df_test.columns)
for col in missing_cols:
    df_test[col] = 0  # Missing columns in the test set get filled with 0
df_test = df_test[df_train.columns]

# Separate input and target variables
X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

# Which Features have the most influence?
mi_scores = mutual_info_regression(X_train, y_train)
mi_scores = pd.Series(mi_scores, index=X_train.columns)
mi_scores.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6), title="Features that contribute the most to the models")
plt.show()
# Power_kw_lag1 and hour have the most influence, then Solar rad., cos_hour, Temperature

corr_matrix = X_train.corr()
# Heatmap of the correlation
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Feature-Korrelationen der wichtigsten Features")
plt.show()

# Selecting the most influential features based on the mutual information analysis
important_features = [ 'Power_kW_lag1','hour', 'cos_hour', 'solarRad_W/m2', 'temp_C' ]

# Update the training and testing datasets to include only the important features
X_train_important = X_train[important_features]
X_test_important = X_test[important_features]

# Feature Scaling (Only on the important features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_important)
X_test_scaled = scaler.transform(X_test_important)

# Feature-Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_important)
X_test_scaled = scaler.transform(X_test_important)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
#Evaluation
print("Linear Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Actual vs Predicted Power Consumption Plot Linear Regression
plt.figure(figsize=(12, 6))
plt.plot(df_test["datetime"], y_test, label="Actual Consumption (kW)", color='blue')
plt.plot(df_test["datetime"], y_pred_lr, label="Predicted Power (kW)", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("Actual vs Predicted Power Consumption with Linear Regression")
plt.xlim(df_test["datetime"].min(), df_test["datetime"].max())
plt.legend()
plt.grid()
plt.show()

# Scatter Plot of Actual vs Predicted Consumption with Regression Line
plt.figure(figsize=(8, 8))
sns.regplot(x=y_test, y=y_pred_lr, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel("Actual Consumption (kW)")
plt.ylabel("Predicted Consumption (kW)")
plt.title("Actual vs. Predicted Power Consumption with Linear Regression")
plt.grid()
plt.show()


# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
# Evaluation
print("Gradient Boosting Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_gb))
print("MSE:", mean_squared_error(y_test, y_pred_gb))
print("R2 Score:", r2_score(y_test, y_pred_gb))

# Plot actual vs predicted values Boosting Regression
plt.figure(figsize=(12, 6))
plt.plot(df_test["datetime"], y_test, label="Actual Power (kW)", color='blue')
plt.plot(df_test["datetime"], y_pred_gb, label="Predicted Power (kW)", color='green', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("Actual vs Predicted Power Consumption with Gradient Boosting Regression")
plt.xlim(df_test["datetime"].min(), df_test["datetime"].max())
plt.legend()
plt.grid()
plt.show()

# Scatter Plot of Actual vs Predicted Consumption with Regression Line
plt.figure(figsize=(8, 8))
sns.regplot(x=y_test, y=y_pred_gb, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel("Actual Consumption (kW)")
plt.ylabel("Predicted Consumption (kW)")
plt.title("Actual vs. Predicted Power Consumption with Boosting Regression")
plt.grid()
plt.show()
#The line chart gives a good impression of the general course of the predictions, 
#but the scatter chart shows clearer errors and distortions.

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
# Evaluation
print("Random Forest Regression:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))


# Actual vs Predicted Power Consumption Plot
plt.figure(figsize=(12, 6))
plt.plot(df_test["datetime"], y_test, label="Actual Power (kW)", color='blue')
plt.plot(df_test["datetime"], y_pred_rf, label="Predicted Power (kW)", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("Actual vs Predicted Power Consumption with random forest regression")
plt.legend()
plt.xlim(df_test["datetime"].min(), df_test["datetime"].max())
plt.grid()
plt.show()

# Scatter Plot of Actual vs Predicted Consumption with Regression Line
plt.figure(figsize=(8, 8))
sns.regplot(x=y_test, y=y_pred_rf, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel("Actual Consumption (kW)")
plt.ylabel("Predicted Consumption (kW)")
plt.title("Actual vs. Predicted Power Consumption with random forest regression")
plt.grid()
plt.show()
#Both plots from the random forest regression are loocking better than the other ones
#and this regression model has the best Evaluation -> thats why I choosed this one 

#Saving the Regression Modells in the folder
project_folder = "Project 1 Energy Services IST1115467"
# Is there really a folder called Project 1 Energy Services IST1115467
if not os.path.exists(project_folder):
    os.makedirs(project_folder) #If the folder is not existing, create one
    
#I first did the saving just for the random forest modell
pickle_file = os.path.join(project_folder, "random_forest_model.pkl")
# save Random Forest Regression Modell
with open(pickle_file, "wb") as f:
    pickle.dump(rf_model, f)
print("Random Forest Modell saved.")


# Then I changed the code to save all models: Paths for saving the models
#pickle_files = {
#    "random_forest_model.pkl": rf_model,
#    "linear_regression_model.pkl": lr_model,
#    "boosting_model.pkl": gb_model
#}
# Save all models
#for filename, model in pickle_files.items():
#    if model is not None:
#        file_path = os.path.join(project_folder, filename)
#        with open(file_path, "wb") as f:
#            pickle.dump(model, f)
#        print(f"Modell saved")

#save the data to calculate in the second part the log1 values for the 2019 data 
df_merged.to_csv(r'C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\Merged_Data.csv', index=False) 
