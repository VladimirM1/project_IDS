import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns


data = pd.read_csv("data/first_dataset.csv")

features = ['danceability', 'energy', 'key', 'valence', 'tempo']
target = 'popularity'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train_scaled, y_train)

rf_predictions = rf_model.predict(X_test_scaled)

rf_mae = mean_absolute_error(y_test, rf_predictions)

print(f'Random Forest Mean Absolute Error: {rf_mae}')

new_song_features = [[0.8, 0.8, 2, 0.9, 100.0]]
new_song_features_scaled = scaler.transform(new_song_features)
rf_future_popularity = rf_model.predict(new_song_features_scaled)

print(f'Random Forest Predicted Future Popularity: {rf_future_popularity[0]}')


linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)
linear_mae = mean_absolute_error(y_test, linear_predictions)

print(f'Linear Regression Mean Absolute Error: {linear_mae}')

linear_future_popularity = linear_model.predict(new_song_features_scaled)

print(f'Linear Regression Predicted Future Popularity: {linear_future_popularity[0]}')

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

gb_model.fit(X_train_scaled, y_train)

gb_predictions = gb_model.predict(X_test_scaled)

gb_mae = mean_absolute_error(y_test, gb_predictions)

print(f'Gradient Boosting Mean Absolute Error: {gb_mae}')

gb_future_popularity = gb_model.predict(new_song_features_scaled)

print(f'Gradient Boosting Predicted Future Popularity: {gb_future_popularity[0]}')

models = ['Random Forest', 'Linear Regression', 'Gradient Boosting']
predicted_popularity = [rf_future_popularity[0], linear_future_popularity[0], gb_future_popularity[0]]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=predicted_popularity, palette='viridis')
plt.title('Predicted Future Popularity from Different Models')
plt.ylabel('Predicted Popularity')
plt.savefig("goal3_bar.pdf")
plt.show()