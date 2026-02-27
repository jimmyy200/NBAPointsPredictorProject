import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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

#absolute_path = os.path.abspath(filename)

#print(absolute_path)


df = pd.read_csv('nba_master_dataset.csv')



# Data cleaning, only look at stats where the player played over 0 minutes
df['numMinutes'] = pd.to_numeric(df['numMinutes'], errors='coerce')
df['numMinutes'] = df['numMinutes'].fillna(0)
df = df[df['numMinutes'] > 0]

# 5 games averages
df['pts_rolling_5'] = df.groupby('personId')['points'].transform(lambda x: x.rolling(window=5, closed='left').mean())


# drop not a number for 5 game avarages and points
df = df.dropna(subset=['pts_rolling_5'])

features = ['pts_rolling_5']
X = df[features]
y = df['points']

# training for 80% of the data and it tests itself on the rest of the 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Mean absolute error tells us exactly how much the model can expect to be off by
# MAE is currently 4.61 points off, meaning that if the model predicts the player to score 25 points, the player should be expected to score 20-30 points
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} points")


coeff_df = pd.DataFrame(model.coef_, features, columns=['Coefficient'])

# print out pts_rolling_5
# if output for that is equal to 1, model predicts the player to score exactly the same amount of pounts as their last 5 games
# The model prints out 0.858 for pts_rolling_5 which means that it predicts the player to slightly shoot worse than their last 5 games
print("\nFeature Importance:")
print(coeff_df) 

# combining first name and last name
df['full_name'] = df['firstName'].str.strip().str.lower() + ' ' + df['lastName'].str.strip().str.lower()

def predict_player_points(player_name, df, model):
    search_name = player_name.strip().lower()

    player_data = df[df['full_name'] == search_name].copy()

    
    if player_data.empty:
        return f"Player '{player_name}' not found in dataset."

    #take most recent game for most up to date stats
    latest_game = player_data.iloc[0]
    
    
    current_features = np.array([[latest_game['pts_rolling_5']]])

    prediction = model.predict(current_features)
    
    return prediction[0]

# individual player prediction
name = "Luka Doncic"
pred = predict_player_points(name, df, model)
print(f"Predicted points for {name} in his next game: {pred:.1f}")


while (True):
    inp = input("Enter A NBA Player Name (enter 'exit' to exit): ")
    if (inp == "exit"):
        break
    pred = predict_player_points(inp, df, model)
    print(f"Predicted points for {inp} in his next game: {pred:.1f}")   