import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime

plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Times')
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('lines', markersize=10)
plt.rc('lines', linewidth=2)
plt.rc('figure', titlesize=18)
plt.rc('figure', figsize=(15, 10))


def get_weather_data(year, month):
    url = f'https://www.timeanddate.com/weather/usa/new-york/historic?month={month}&year={year}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    avg_temps = []
    precipitation = []

    weather_table = soup.find('table', class_='zebra fw tb-wt zebra va-m')
    if not weather_table:
        return avg_temps, precipitation

    rows = weather_table.find_all('tr')[1:]

    for row in rows:
        cells = row.find_all('td')
        avg_temp = cells[2].text.split('/')[0].strip()
        precip = cells[3].text.strip()

        avg_temps.append(float(avg_temp))
        precipitation.append(float(precip.replace('in', '')))

    return avg_temps, precipitation


def plot_weather_data(dataframe):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Temperature (°F)', color='tab:red')
    ax1.plot(dataframe['Month'], dataframe['Average Temperature'], color='tab:red', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Precipitation (in)', color='tab:blue')
    ax2.plot(dataframe['Month'], dataframe['Precipitation'], color='tab:blue', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title('New York City Weather Data (2021)')
    plt.show()


# Scrape weather data for each month in 2021
weather_data = []

for month in range(1, 13):
    avg_temps, precipitation = get_weather_data(2022, month)
    if not avg_temps or not precipitation:
        print(f"Could not fetch weather data for month {month}. Skipping...")
        continue
    weather_data.append((month, sum(avg_temps) / len(avg_temps), sum(precipitation)))

# Create a DataFrame from the scraped data
columns = ['Month', 'Average Temperature', 'Precipitation']
weather_df = pd.DataFrame(weather_data, columns=columns)

# Plot the weather data
plot_weather_data(weather_df)
