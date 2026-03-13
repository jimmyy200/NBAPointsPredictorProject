import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#  GitHub: https://github.com/jimmyy200/NBAPointsPredictorProject/tree/main (Dataset files are in there)


df = pd.read_csv('C:/Users/jimmy_p2fa2zj/Documents/GitHub/NBAPointsPredictorProject/nba_master_dataset.csv')



# Data cleaning, only look at stats where the player played over 0 minutes
df['numMinutes'] = pd.to_numeric(df['numMinutes'], errors='coerce')
df['numMinutes'] = df['numMinutes'].fillna(0)
df = df[df['numMinutes'] > 0]

# 10 games averages
df['pts_rolling_window'] = df.groupby('personId')['points'].transform(lambda x: x.rolling(window=10, closed='left').mean())


# drop not a number for 10 game avarages and points
df = df.dropna(subset=['pts_rolling_window'])

features = ['pts_rolling_window', 'assists', 'fieldGoalsAttempted', 'fieldGoalsMade', 'freeThrowsAttempted', 'freeThrowsMade']
X = df[features]
y = df['points']

# training for 80% of the data and it tests itself on the rest of the 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Mean absolute error tells us exactly how much the model can expect to be off by
# MAE is currently 0.67 points off so predicted values are pretty close to actual values
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} points")

# combining first name and last name
df['full_name'] = df['firstName'].str.strip().str.lower() + ' ' + df['lastName'].str.strip().str.lower()

def predict_player_points(player_name, df, model):
    search_name = player_name.strip().lower()

    player_data = df[df['full_name'] == search_name].copy()

    
    if player_data.empty:
        return f"Player '{player_name}' not found in dataset."

    #take most recent game for most up to date stats
    latest_game = player_data.iloc[0]
    
    
    current_features = latest_game[['pts_rolling_window', 'assists', 'fieldGoalsAttempted', 'fieldGoalsMade', 'freeThrowsAttempted', 'freeThrowsMade']].fillna(0).values.reshape(1, -1)


    prediction = model.predict(current_features)
    
    return prediction[0]



# individual player prediction
print("Sample Prediction \n")
name = "Luka Doncic"
pred = predict_player_points(name, df, model)
print(f"Predicted points for {name} in his next game: {pred:.1f}")


while (True):
    inp = input("Enter A NBA Player Name (enter 'exit' to exit): ")
    if (inp == "exit"):
        break
    pred = predict_player_points(inp, df, model)
    print(f"Predicted points for {inp} in his next game: {pred:.1f}")   
