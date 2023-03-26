import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import matplotlib.pyplot as plt
import schedule
import time
import os
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def scrape_data(chrome_driver_path="path/to/chromedriver"):
    url = "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html"

    # Configure the WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode to avoid opening a browser window
    options.add_argument("--disable-gpu")  # May be necessary for compatibility with some systems
    driver = webdriver.Chrome(executable_path=chrome_driver_path, options=options)
    # Load the URL
    driver.get(url)
    # Wait for the exchange rate table to load and locate the USD/EUR exchange rate
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".forextable"))
        )
        usd_eur_rate = None
        for row in driver.find_elements(By.CSS_SELECTOR, ".forextable tr"):
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) == 3 and cells[0].text.strip() == "US dollar":
                usd_eur_rate = float(cells[2].text.strip())
                break
    finally:
        driver.quit()
    if usd_eur_rate is None:
        raise ValueError("Could not find the USD/EUR exchange rate")
    return usd_eur_rate


"""def scrape_data():
    url = "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Locate the data in the HTML and extract the USD/EUR exchange rate
    data = None
    for row in soup.find("table", class_="forextable").find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 3 and cells[0].text.strip() == "US dollar":
            data = float(cells[2].text.strip())
            break
    if data is None:
        raise ValueError("Could not find the USD/EUR exchange rate")
    return data"""


def save_to_csv(data, filename="data.csv"):
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "USD/EUR"])  # Column headers
        writer.writerow([pd.Timestamp.now(), data])


def process_and_plot_data(filename="data.csv"):
    df = pd.read_csv(filename)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(df["Timestamp"], df["USD/EUR"], label="USD/EUR")
    plt.xlabel("Timestamp")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.title("USD/EUR Exchange Rate")
    plt.show()


def main_task():
    data = scrape_data()
    save_to_csv(data)
    process_and_plot_data()


# Schedule the script to run at 8:00 GMT every day
schedule.every().second.do(main_task)
# schedule.every().day.at("08:00").do(main_task)
# schedule.every().monday.at("08:00").do(main_task)

# Keep the script running indefinitely
while True:
    schedule.run_pending()
    time.sleep(1)
