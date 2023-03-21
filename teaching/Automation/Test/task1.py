# task1.py

import pandas as pd
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_programming_languages_by_type"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'class': 'wikitable'})
df = pd.read_html(str(table))[0]

# Clean up the DataFrame
df.columns = df.iloc[0]
df = df.drop(0)

# Save the DataFrame to a CSV file
df.to_csv('programming_languages.csv', index=False)
