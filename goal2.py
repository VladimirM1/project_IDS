import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/first_dataset.csv")

top_songs_global = data[data['country'].isnull()].groupby('name')['popularity'].mean().nlargest(10).index
top_songs_data = data[data['name'].isin(top_songs_global)]

plt.figure(figsize=(14, 8))
sns.lineplot(data=top_songs_data, x='snapshot_date', y='daily_rank', hue='artists', marker='o')

plt.title('Change in Rankings of Top 10 Global Songs Over Time')
plt.xlabel('Snapshot Date')
plt.ylabel('Daily Rank')

plt.legend(title='Artists', bbox_to_anchor=(1, 1), loc='upper left', fontsize='large')

plt.yticks(range(0, 50, 10))

plt.tight_layout()
plt.savefig("goal2_dailyrankChange.pdf")

plt.show()

song_characteristics = ['danceability', 'energy', 'tempo', 'daily_rank', 'popularity','loudness','duration_ms']

corr_matrix = top_songs_data[song_characteristics].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=song_characteristics, yticklabels=song_characteristics)
plt.title('Correlation Between Song Characteristics and Rankings for Top 10 Songs')
plt.savefig("goal2_heatmap.pdf")
plt.show()
