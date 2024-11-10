
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "farm_fishing_data_time_series_smooth_target.csv"
data = pd.read_csv(file_path)

# Step 1: Preview the data
print("First five rows of the dataset:")
print(data.head())

print("\nData Info:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())

# Step 2: Statistical Summary
print("\nStatistical Summary:")
print(data.describe())

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Step 3: Calculate correlations between Yield and other numerical features
correlations = data.corr()["Yield (kg)"].sort_values(ascending=False)
print("\nCorrelation with Yield:")
print(correlations)

# Step 4: Visualize the Yield time series trend
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Yield (kg)'])
plt.title('Yield (kg) Over Time')
plt.xlabel('Date')
plt.ylabel('Yield (kg)')
plt.show()
