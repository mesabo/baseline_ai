#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/28/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import numpy as np
import pandas as pd

def generate_time_series_data_with_yield(
    start_date="2020-01-01",
    end_date="2024-12-31",
    seasonal_amplitude=10,
    trend_slope=0.01,
    noise_level=5,
    include_seasonality=True,
    include_trend=True
):
    """
    Generates a synthetic time-series dataset with `Yield` as the target variable
    and incorporates natural connections among features.

    Parameters:
        start_date (str): Starting date of the time series.
        end_date (str): Ending date of the time series.
        seasonal_amplitude (float): Amplitude of seasonal variation.
        trend_slope (float): Slope of the linear trend.
        noise_level (float): Standard deviation of random noise.
        include_seasonality (bool): Whether to include seasonal variation.
        include_trend (bool): Whether to include a linear trend.

    Returns:
        pd.DataFrame: Generated dataset with `Yield` as a calculated target variable.
    """
    # Generate a date range
    dates = pd.date_range(start=start_date, end=end_date)

    # Number of rows
    num_rows = len(dates)

    # Generate base environmental factors
    sunlight_hours = 12 + 2 * np.sin(2 * np.pi * dates.dayofyear / 365)  # Seasonal sunlight
    wind_speed = 5 + 2 * np.cos(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 1.5, num_rows)  # Seasonal wind
    water_temperature = 20 + 5 * np.sin(2 * np.pi * dates.dayofyear / 365) + 0.1 * sunlight_hours - 0.2 * wind_speed
    water_clarity = 1.5 - 0.05 * wind_speed + 0.1 * sunlight_hours + np.random.normal(0, 0.1, num_rows)  # Turbidity proxy

    # Chemical factors
    dissolved_oxygen = 8 + 0.3 * water_temperature - 0.1 * wind_speed + 0.2 * water_clarity + np.random.normal(0, 1, num_rows)
    ammonia_level = 2 + 0.5 * sunlight_hours - 0.2 * dissolved_oxygen + 0.1 * wind_speed
    nitrate_level = 30 + 0.4 * ammonia_level - 0.3 * dissolved_oxygen + 0.2 * water_clarity

    # Water quality affecting pH level
    ph_level = 7 - 0.1 * nitrate_level + 0.2 * dissolved_oxygen + 0.05 * water_clarity + np.random.uniform(0, 0.3, num_rows)

    # Fish-related factors
    fish_density = 5 + 0.3 * dissolved_oxygen - 0.1 * ammonia_level - 0.05 * wind_speed
    feed_rate = 1 + 0.5 * fish_density - 0.1 * wind_speed + np.random.uniform(0, 0.3, num_rows)

    # Yield based on key predictors
    seasonal = (
        seasonal_amplitude * np.sin(2 * np.pi * dates.dayofyear / 365)
        if include_seasonality
        else 0
    )
    trend = trend_slope * np.arange(num_rows) if include_trend else 0
    noise = np.random.normal(0, noise_level, num_rows)

    yield_values = (
        50
        + 0.5 * water_temperature
        - 1.2 * ph_level
        + 0.8 * dissolved_oxygen
        - 0.3 * ammonia_level
        + 0.1 * nitrate_level
        + 0.7 * fish_density
        + 0.2 * feed_rate
        + seasonal
        + trend
        + noise
    )

    # Combine into a DataFrame
    data = pd.DataFrame({
        "Date": dates,
        "Sunlight Hours": sunlight_hours.round(2),
        "Wind Speed (km/h)": wind_speed.round(2),
        "Water Temperature (°C)": water_temperature.round(2),
        "Water Clarity (index)": water_clarity.round(2),
        "Dissolved Oxygen (mg/L)": dissolved_oxygen.round(2),
        "Ammonia Level (mg/L)": ammonia_level.round(2),
        "Nitrate Level (mg/L)": nitrate_level.round(2),
        "pH Level": ph_level.round(2),
        "Fish Density (fish/m³)": fish_density.round(2),
        "Feed Rate (kg/day)": feed_rate.round(2),
        "Yield (kg)": yield_values.round(2)
    })

    return data
def save_dataset(data, file_path):
    """
    Save the generated dataset to a CSV file with only column headers in the header.

    Parameters:
        data (pd.DataFrame): The dataset to save.
        file_path (str): Path to the output CSV file.
    """
    if data.empty:
        print("Warning: The dataset is empty. No file was saved.")
    else:
        data = data.round(2)
        data.to_csv(file_path, index=False, header=True)
        print(f"Dataset saved successfully to {file_path}")

def main():
    """
    Main function to generate the dataset and save it.
    """
    # Configuration parameters
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    seasonal_amplitude = 15
    trend_slope = 0.02
    noise_level = 8
    include_seasonality = True
    include_trend = True
    output_file = "./synthetic_fishing_data.csv"

    # Generate synthetic dataset
    dataset = generate_time_series_data_with_yield(
        start_date=start_date,
        end_date=end_date,
        seasonal_amplitude=seasonal_amplitude,
        trend_slope=trend_slope,
        noise_level=noise_level,
        include_seasonality=include_seasonality,
        include_trend=include_trend
    )

    # Save the dataset
    save_dataset(dataset, output_file)

if __name__ == "__main__":
    main()