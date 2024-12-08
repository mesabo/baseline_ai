#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/08/2024
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import pandas as pd
def load_csv_data(filepath):
    """
    Load CSV data from the specified file path.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(filepath)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns, size of {df.size} ")
    #print(df.head())
    return df
