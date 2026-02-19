import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd

#players_csv = pd.read_csv('C:/Users/jimmy_p2fa2zj/Documents/GitHub/NBAPointsPredictorProject/Players.csv', low_memory=False)
#games_csv = pd.read_csv('C:/Users/jimmy_p2fa2zj/Documents/GitHub/NBAPointsPredictorProject/Games.csv', low_memory=False)
#player_stats_csv = pd.read_csv('C:/Users/jimmy_p2fa2zj/Documents/GitHub/NBAPointsPredictorProject/PlayerStatistics.csv', low_memory=False)

#print("Games Columns:", players_csv.columns.tolist())

"""
master_df = pd.merge(player_stats_csv, games_csv[['gameId', 'gameDateTimeEst', 'hometeamId', 'awayteamId']], on='gameId', how='left')

master_df = pd.merge(master_df, players_csv[['personId', 'heightInches', 'bodyWeightLbs']], on='personId', how='left')

master_df.to_csv('nba_master_dataset.csv', index=False)

print("Master file created successfully!")
"""

filename = 'nba_master_dataset.csv'

absolute_path = os.path.abspath(filename)

print(absolute_path)

"""
df = pd.read_csv('nba_master_dataset.csv')

print(df.head())

print (df.info())
"""