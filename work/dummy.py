import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data
np.random.seed(0)
filenames = np.random.choice(['file1', 'file2', 'file3', "file4"], size=100)
users = np.random.choice(['user1', 'user2', 'user3'], size=100)
years = np.random.randint(2010, 2015, size=100)

# Create DataFrame
df = pd.DataFrame({'filename': filenames, 'user': users, 'year': years})

# Group by year and filename, count unique users
user_counts = df.groupby(['year', 'filename'])['user'].nunique().unstack().fillna(0)
user_counts.plot.scatter(x='year', y='filename', s=100, alpha=0.5)
plt.xlabel('Group Column 1')
plt.ylabel('Group Column 2')
plt.title('Count of Users')
plt.show()
exit()

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.imshow(user_counts, cmap='viridis', aspect='auto', interpolation='nearest')

# Add annotations and customize markers
for i in range(len(user_counts.index)):
    for j in range(len(user_counts.columns)):
        plt.text(j, i, user_counts.iloc[i, j], ha='center', va='center', color='black')

# Add color bar
plt.colorbar(label='Number of Users')

# Set labels and title
plt.xlabel('Filename')
plt.ylabel('Year')
plt.title('Number of Users per Year and Filename')

# Set x-axis and y-axis ticks
plt.xticks(np.arange(len(user_counts.columns)), user_counts.columns, rotation=90)
plt.yticks(np.arange(len(user_counts.index)), user_counts.index)

# Show plot
plt.tight_layout()
plt.show()
