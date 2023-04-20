import requests
import schedule
import time
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

url = 'https://analytics.google.com/analytics/web/#/my-reports/33vbyo_yRY6G11yuRDlbNg/a54516992w87479473p92320289/44-table.plotKeys=%5B%5D&44-table.rowStart=0&44-table.rowCount=5000'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
table = soup.find('table', {'class': '_GAw'})  # find the table element by class name
rows = table.find_all('tr')  # find all the rows of the table

data = []
for row in rows:
    cells = row.find_all('td')
    row_data = [cell.text.strip() for cell in cells]
    data.append(row_data)

df = pd.DataFrame(data[1:], columns=data[0])  # create a Pandas dataframe with the table data
print(df.head())
