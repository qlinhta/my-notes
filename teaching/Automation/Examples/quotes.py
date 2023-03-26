import requests
import csv
import random


def get_quotes():
    url = "https://type.fit/api/quotes"
    response = requests.get(url)
    quotes_data = response.json()
    return quotes_data


def save_to_csv(quotes, filename="quotes.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["quote", "author"])
        for quote in quotes:
            writer.writerow([quote["text"], quote["author"]])


def main():
    all_quotes = get_quotes()
    unique_quotes = random.sample(all_quotes, 100)

    save_to_csv(unique_quotes)


if __name__ == "__main__":
    main()
