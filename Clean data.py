#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from meteostat import Point, Daily,Hourly
import numpy as np


# In[2]:


# Function to retrieve data for a specific date from ThingSpeak-box1
def get_data_for_date(date):
    url = "https://api.thingspeak.com/channels/2530004/feeds.json"
    params = {
        "start": date.strftime("%Y-%m-%d"),
        "end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "Europe/Berlin",  # Adjust timezone as per your location
        "round": 2,  # Round to 2 decimal places
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["feeds"]
        return pd.DataFrame(data)
    else:
        print(f"Failed to retrieve data for {date.strftime('%Y-%m-%d')}")
        return pd.DataFrame()

# Define the start and end dates for your data retrieval
start_date = datetime(2024, 5, 21)
end_date = datetime.now()

# Iterate over the range of dates and retrieve data
all_data = []
current_date = start_date
while current_date <= end_date:
    data_for_date = get_data_for_date(current_date)
    all_data.append(data_for_date)
    current_date += timedelta(days=1)

# Concatenate all the data into a single DataFrame
df1 = pd.concat(all_data, ignore_index=True)

# Rename columns
df1 = df1.rename(columns={'field1': 'humidity', 'field2': 'temp', 'field3': 'lux', 'field4': 'Noise', 'field5': 'CO2'})

# Convert columns to appropriate data types
df1['created_at'] = pd.to_datetime(df1['created_at'])
df1['humidity'] = df1['humidity'].astype(float)
df1['temp'] = df1['temp'].astype(float)
df1['lux'] = df1['lux'].astype(float)
df1['Noise'] = df1['Noise'].astype(float)
df1['CO2'] = df1['CO2'].astype(float)

# Display the DataFrame
print(df1.head())


# In[3]:


def get_data_for_date(date):
    url = "https://api.thingspeak.com/channels/2530008/feeds.json"
    params = {
        "start": date.strftime("%Y-%m-%d"),
        "end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "Europe/Berlin",  # Adjust timezone as per your location
        "round": 2,  # Round to 2 decimal places
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["feeds"]
        return pd.DataFrame(data)
    else:
        print(f"Failed to retrieve data for {date.strftime('%Y-%m-%d')}")
        return pd.DataFrame()

# Define the start and end dates for your data retrieval
start_date = datetime(2024, 5, 21)
end_date = datetime.now()

# Iterate over the range of dates and retrieve data
all_data = []
current_date = start_date
while current_date <= end_date:
    data_for_date = get_data_for_date(current_date)
    all_data.append(data_for_date)
    current_date += timedelta(days=1)

# Concatenate all the data into a single DataFrame
df2 = pd.concat(all_data, ignore_index=True)

# Rename columns
df2 = df2.rename(columns={'field1': 'humidity', 'field2': 'temp', 'field3': 'lux', 'field4': 'Noise', 'field5': 'CO2'})

# Convert columns to appropriate data types
df2['created_at'] = pd.to_datetime(df2['created_at'])
df2['humidity'] = df2['humidity'].astype(float)
df2['temp'] = df2['temp'].astype(float)
df2['lux'] = df2['lux'].astype(float)
df2['Noise'] = df2['Noise'].astype(float)
df2['CO2'] = df2['CO2'].astype(float)

# Display the DataFrame
print(df2.head())


# In[4]:


def get_data_for_date(date):
    url = "https://api.thingspeak.com/channels/2530008/feeds.json"
    params = {
        "start": date.strftime("%Y-%m-%d"),
        "end": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "Europe/Berlin",  # Adjust timezone as per your location
        "round": 2,  # Round to 2 decimal places
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()["feeds"]
        return pd.DataFrame(data)
    else:
        print(f"Failed to retrieve data for {date.strftime('%Y-%m-%d')}")
        return pd.DataFrame()

# Define the start and end dates for your data retrieval
start_date = datetime(2024, 5, 21)
end_date = datetime.now()

# Iterate over the range of dates and retrieve data
all_data = []
current_date = start_date
while current_date <= end_date:
    data_for_date = get_data_for_date(current_date)
    all_data.append(data_for_date)
    current_date += timedelta(days=1)

# Concatenate all the data into a single DataFrame
df4 = pd.concat(all_data, ignore_index=True)

# Rename columns
df4 = df4.rename(columns={'field1': 'temp', 'field2':'humidity', 'field3': 'lux', 'field4':'Noise', 'field5':'CO2'})

# Convert columns to appropriate data types
df4['created_at'] = pd.to_datetime(df4['created_at'])
df4['humidity'] = df4['humidity'].astype(float)
df4['temp'] = df4['temp'].astype(float)
df4['lux'] = df4['lux'].astype(float)
df4['Noise'] = df4['Noise'].astype(float)
df4['CO2'] = df4['CO2'].astype(float)

# Display the DataFrame
print(df4.head())


# In[5]:


# Check for NaN values
print("Checking for NaN values in df1:")
print(df1.isna().sum())
print("\nChecking for NaN values in df2:")
print(df2.isna().sum())
print("\nChecking for NaN values in df4:")
print(df4.isna().sum())

# Remove rows with NaN values and reassign to the original data frames
df1 = df1.dropna()
df2 = df2.dropna()
df4 = df4.dropna()

# Verify that NaN values are removed
print("\nAfter removing NaN values:")
print("df1 shape:", df1.shape)
print("df2 shape:", df2.shape)
print("df4 shape:", df4.shape)


# In[6]:


df1.dtypes


# In[7]:


df2.dtypes


# In[8]:


df4.dtypes


# In[10]:


# outlier removal using Z-score
from scipy import stats

#calculate z-scores
z_scores_df1 = np.abs(stats.zscore(df1[['humidity', 'temp', 'lux', 'Noise', 'CO2']]))
z_scores_df2 = np.abs(stats.zscore(df2[['humidity', 'temp', 'lux', 'Noise', 'CO2']]))
z_scores_df4 = np.abs(stats.zscore(df4[['humidity', 'temp', 'lux', 'Noise', 'CO2']]))



# In[11]:


#identify outliers
outliers_df1 = df1[(z_scores_df1 > 3).any(axis=1)]
outliers_df2 = df2[(z_scores_df2 > 3).any(axis=1)]
outliers_df4 = df4[(z_scores_df4 > 3).any(axis=1)]


# In[12]:


#count outliers
num_outliers_df1 = outliers_df1.shape[0]
num_outliers_df2 = outliers_df2.shape[0]
num_outliers_df4 = outliers_df4.shape[0]

print(f"Number of outliers in df1: {num_outliers_df1}")
print(f"Number of outliers in df2: {num_outliers_df2}")
print(f"Number of outliers in df4: {num_outliers_df4}")


# In[13]:


#examine removed outliers
print("Outliers removed from df1:")
display(outliers_df1)

print("Outliers removed from df2:")
display(outliers_df2)

print("Outliers removed from df4:")
display(outliers_df4)


# In[14]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df2['created_at'], df2['CO2'], label='After Outlier Removal', color='blue', alpha=0.6)
plt.scatter(outliers_df2['created_at'], outliers_df2['CO2'], label='Outliers Removed', color='red', alpha=0.6)
plt.xlabel('Timestamp')
plt.ylabel('CO2 Levels')
plt.legend()
plt.title('CO2 Levels Before and After Outlier Removal')
plt.show()


# In[15]:


#timebased smoothing
# Resampling and smoothing function
def resample_and_smooth(df, time_col='created_at', freq='30T', window=3):
    # Ensure the datetime column is set as the index
    df = df.set_index(time_col)
    
    # Resample to 30-minute intervals
    df_resampled = df.resample(freq).mean()
    
    # Apply a rolling average with the specified window
    df_smoothed = df_resampled.rolling(window=window, min_periods=1).mean()
    
    # Reset the index to get the datetime back as a column
    df_smoothed = df_smoothed.reset_index()
    
    return df_smoothed

# Apply to all three dataframes
df1_smoothed = resample_and_smooth(df1)
df2_smoothed = resample_and_smooth(df2)
df4_smoothed = resample_and_smooth(df4)


# In[16]:


# Function to print statistics
def print_statistics(df, df_name):
    print(f"Statistics for {df_name}:")
    print(df.describe())
    print("\n")

# Apply to all smoothed dataframes
print_statistics(df1_smoothed, "df1_smoothed")
print_statistics(df2_smoothed, "df2_smoothed")
print_statistics(df4_smoothed, "df4_smoothed")


# In[17]:


import matplotlib.pyplot as plt

# Time Series Plot for CO2, Temp, and Humidity
plt.figure(figsize=(15, 10))

# CO2 Levels
plt.subplot(3, 1, 1)
plt.plot(df1_smoothed['created_at'], df1_smoothed['CO2'], label='df1 CO2', color='blue')
plt.plot(df2_smoothed['created_at'], df2_smoothed['CO2'], label='df2 CO2', color='green')
plt.plot(df4_smoothed['created_at'], df4_smoothed['CO2'], label='df4 CO2', color='red')
plt.title('CO2 Levels Over Time')
plt.xlabel('Time')
plt.ylabel('CO2 (ppm)')
plt.legend()

# Temperature
plt.subplot(3, 1, 2)
plt.plot(df1_smoothed['created_at'], df1_smoothed['temp'], label='df1 Temp', color='blue')
plt.plot(df2_smoothed['created_at'], df2_smoothed['temp'], label='df2 Temp', color='green')
plt.plot(df4_smoothed['created_at'], df4_smoothed['temp'], label='df4 Temp', color='red')
plt.title('Temperature Over Time')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()

# Humidity
plt.subplot(3, 1, 3)
plt.plot(df1_smoothed['created_at'], df1_smoothed['humidity'], label='df1 Humidity', color='blue')
plt.plot(df2_smoothed['created_at'], df2_smoothed['humidity'], label='df2 Humidity', color='green')
plt.plot(df4_smoothed['created_at'], df4_smoothed['humidity'], label='df4 Humidity', color='red')
plt.title('Humidity Over Time')
plt.xlabel('Time')
plt.ylabel('Humidity (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Box Plots
plt.figure(figsize=(15, 6))

# CO2 Boxplot
plt.subplot(1, 3, 1)
plt.boxplot([df1_smoothed['CO2'], df2_smoothed['CO2'], df4_smoothed['CO2']], labels=['df1', 'df2', 'df4'])
plt.title('CO2 Distribution')
plt.ylabel('CO2 (ppm)')

# Temperature Boxplot
plt.subplot(1, 3, 2)
plt.boxplot([df1_smoothed['temp'], df2_smoothed['temp'], df4_smoothed['temp']], labels=['df1', 'df2', 'df4'])
plt.title('Temperature Distribution')
plt.ylabel('Temperature (°C)')

# Humidity Boxplot
plt.subplot(1, 3, 3)
plt.boxplot([df1_smoothed['humidity'], df2_smoothed['humidity'], df4_smoothed['humidity']], labels=['df1', 'df2', 'df4'])
plt.title('Humidity Distribution')
plt.ylabel('Humidity (%)')

plt.tight_layout()
plt.show()


# In[18]:


# Checking if the data has any issues
print(df1_smoothed[['CO2', 'temp', 'humidity']].describe())
print(df2_smoothed[['CO2', 'temp', 'humidity']].describe())
print(df4_smoothed[['CO2', 'temp', 'humidity']].describe())

# Adjusted Box Plot Code
plt.figure(figsize=(15, 6))

# CO2 Boxplot
plt.subplot(1, 3, 1)
plt.boxplot([df1_smoothed['CO2'].dropna(), df2_smoothed['CO2'].dropna(), df4_smoothed['CO2'].dropna()], 
            labels=['df1', 'df2', 'df4'])
plt.title('CO2 Distribution')
plt.ylabel('CO2 (ppm)')

# Temperature Boxplot
plt.subplot(1, 3, 2)
plt.boxplot([df1_smoothed['temp'].dropna(), df2_smoothed['temp'].dropna(), df4_smoothed['temp'].dropna()], 
            labels=['df1', 'df2', 'df4'])
plt.title('Temperature Distribution')
plt.ylabel('Temperature (°C)')

# Humidity Boxplot
plt.subplot(1, 3, 3)
plt.boxplot([df1_smoothed['humidity'].dropna(), df2_smoothed['humidity'].dropna(), df4_smoothed['humidity'].dropna()], 
            labels=['df1', 'df2', 'df4'])
plt.title('Humidity Distribution')
plt.ylabel('Humidity (%)')

plt.tight_layout()
plt.show()


# In[ ]:




