# task2.py

import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv('programming_languages.csv')

# add column name to the dataframe
df.columns = ['Language', 'Designed by', 'First appeared', 'Appeared in', 'Extension']

# Plot the data as a bar chart using the 'Language' column as the x-axis and the 'First appeared' column as the y-axis
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(df['Language'], df['First appeared'])
plt.savefig('programming_languages.pdf')
# Show the plot
plt.show()

""" .bat for windows to run both task1.py and task2.py
@echo off
echo Starting Task 1
python task1.py
echo Task 1 completed

echo Starting Task 2
python task2.py
echo Task 2 completed

echo All tasks completed

"""