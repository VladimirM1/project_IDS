import pandas as pd

df = pd.read_csv("data/first_dataset.csv")

all_combinations = pd.DataFrame([(country, spotify_id) for country in df['country'].unique() for spotify_id in df['spotify_id'].unique()], columns=['country', 'spotify_id'])
merged_df = pd.merge(all_combinations, df, on=['country', 'spotify_id'], how='left')
merged_df['daily_rank'].fillna(51, inplace=True)
average_rank_country_df = merged_df.groupby(['country', 'spotify_id'])['daily_rank'].mean().reset_index()
top_song_indices = average_rank_country_df.groupby('country')['daily_rank'].idxmin()
top_songs_by_country = average_rank_country_df.loc[top_song_indices]
result_df = pd.merge(top_songs_by_country, df, on=['country', 'spotify_id'])

result_df = result_df.drop_duplicates(subset='country')
for country, song_name, artists in zip(result_df['country'], result_df['name'], result_df['artists']):
    print(f"Country: {country}, Top Song: {song_name} by {artists}")