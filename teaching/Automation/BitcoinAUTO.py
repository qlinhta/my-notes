import requests
import schedule
import time
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

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
plt.rc('figure', figsize=(10, 10))


def get_bitcoin_price():
    url = 'https://www.coingecko.com/en/coins/bitcoin'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    price_div = soup.find('span', class_='no-wrap')
    price = price_div.text.strip()
    return price


def log_bitcoin_price():
    current_price = get_bitcoin_price()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"{timestamp} - Current Bitcoin price: {current_price}\n"

    # Write the log message to a file
    with open('bitcoin_price_log.txt', 'a') as log_file:
        """
        The 'a' argument means that we are opening the file in append mode.
        This means that any new data that we write to the file will be added to the end of the file.
        
        a = append mode
        w = write mode
        r = read mode
        """
        log_file.write(log_message)

    # Print the log message to the console
    print(log_message.strip())


# Plot the bitcoin price every 5 seconds
prices = []
timestamps = []


def plot_bitcoin_price():
    current_price = get_bitcoin_price()
    timestamp = datetime.now().strftime('%H:%M:%S')
    prices.append(current_price)
    timestamps.append(timestamp)
    df = pd.DataFrame({'Timestamp': timestamps, 'Price': prices})
    df['Price'] = df['Price'].str.replace('$', '')
    df['Price'] = df['Price'].str.replace(',', '')
    df['Price'] = df['Price'].astype(float)
    df.plot(x='Timestamp', y='Price', kind='line', figsize=(20, 8), color='red', linewidth=2, marker='o')
    plt.title('Bitcoin Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid()
    plt.show()


# Schedule the log_bitcoin_price function to run every minute
schedule.every(1).second.do(log_bitcoin_price)

# Run the scheduled tasks
while True:
    schedule.run_pending()
    plot_bitcoin_price()
    for _ in tqdm(range(1), desc="Waiting for the next update", unit="s"):
        time.sleep(1)
