
# Farm Fishing Data EDA

This document provides an exploratory data analysis (EDA) of the farm fishing dataset using `numpy` and `pandas`.

## 1. Load and Preview the Data

We begin by loading the dataset and checking the first few rows, the data types, and missing values.

```python
import pandas as pd

# Load the dataset
file_path = "farm_fishing_data_time_series_smooth_target.csv"
data = pd.read_csv(file_path)

# Preview the data
print(data.head())
data.info()
print(data.isnull().sum())
```

## 2. Statistical Summary

Next, we review the statistical summary to understand the distribution of numerical columns.

```python
# Statistical summary
print(data.describe())
```

## 3. Convert Date Column

Since the `Date` column is currently an object type, we convert it to `datetime` for better handling in time series analysis.

```python
# Convert Date to datetime format
data['Date'] = pd.to_datetime(data['Date'])
```

## 4. Correlation Analysis

To explore relationships between features, we calculate the correlation of each feature with the `Yield (kg)` column.

```python
# Calculate correlations
correlations = data.corr()["Yield (kg)"].sort_values(ascending=False)
print(correlations)
```

## 5. Visualize the Yield Time Series Trend

We plot `Yield (kg)` over time to observe its trend and seasonality, which helps to understand its suitability for forecasting models.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Yield (kg)'])
plt.title('Yield (kg) Over Time')
plt.xlabel('Date')
plt.ylabel('Yield (kg)')
plt.show()
```
