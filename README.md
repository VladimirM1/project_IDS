# Top Spotify Songs in 73 Countries

Author : Vladimir Makarenkov

### Dataset:

The research leverages a robust dataset sourced from Spotify, capturing daily snapshots of the top songs in 73 countries. Spanning over 29,000 entries, the dataset encompasses a diverse range of attributes, including song details (name, artists, album), daily rankings, geographical information, and musical characteristics (danceability, energy, key, tempo).

### Goal 1:

Identify the top song for each country based on the average daily rank. After that identify genres.

Created a DataFrame with all possible combinations of 'country' and 'spotify_id' from the original dataset. Merged the original dataset with the combinations DataFrame on 'country' and 'spotify_id' using a left join. Filled missing values in the 'daily_rank' column with 51, indicating songs that were not in the top 50. Grouped the merged DataFrame by 'country' and 'spotify_id' and calculated the mean 'daily_rank' for each song in each country. Found the indices of the minimum rank for each country, representing the top song. Created a DataFrame containing the top song for each country. Printed the country, top song name, and artists for each country.

### Goal 2:

Analyze the change in rankings of the top 10 global songs over time.

Identified the top 10 global songs based on the mean 'popularity' and filtered the dataset. Created a line plot to visualize the change in rankings of the top 10 global songs over time. Selected relevant song characteristics and generated a heatmap to show the correlation between these characteristics and rankings.

### Goal 3:

Predict the future popularity of songs using machine learning models.
Selected features ('danceability', 'energy', 'key', 'valence', 'tempo') and the target variable ('popularity'). Splitted the data into training and testing sets. Standardized features using StandardScaler. Created a Random Forest Regressor, fitted the model, and evaluated its performance. Created a Linear Regression model, fitted the model, and evaluated its performance. Created a Gradient Boosting Regressor, fitted the model, and evaluated its performance. Predicted the future popularity for a new song using each model. Created a bar plot comparing the predicted future popularity from different models.

Codes can be executed individually, the right order is goal1.py, goal2.py goal3.py. All plots are saved in projects directory aswell with the name of a file of origin in it.

