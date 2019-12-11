# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

data = pd.read_csv("../Datasets/insurance.csv")

print(data.head(15))


count_nan = data.isnull().sum()
print(count_nan[count_nan>0])

data['bmi'].fillna(data['bmi'].mean(), inplace=True)

count_nan = data.isnull().sum()
print(count_nan[count_nan>0])