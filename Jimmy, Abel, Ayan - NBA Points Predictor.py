import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd

file_path = 'C:/Users/jimmy_p2fa2zj/Documents/GitHub/NBAPointsPredictorProject/player.csv'

df = pd.read_csv(file_path)

print(df.head())
df.info()