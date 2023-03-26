import requests
import csv


def get_weather(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    weather_data = response.json()

    if response.status_code != 200:
        print(f"Error fetching weather data: {weather_data['message']}")
        return None

    return {
        "city": city,
        "country": weather_data["sys"]["country"],
        "temperature": weather_data["main"]["temp"],
        "humidity": weather_data["main"]["humidity"],
        "weather": weather_data["weather"][0]["description"],
    }


def save_to_csv(weather_data, filename="weather.csv"):
    if weather_data is None:
        print("No weather data to save")
        return

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["city", "country", "temperature", "humidity", "weather"])
        writer.writerow(
            [weather_data["city"], weather_data["country"], weather_data["temperature"], weather_data["humidity"],
             weather_data["weather"]])


def main():
    api_key = "3dc277b8ab7a5ad9c3de66c540df2d8d"
    city = "London"
    weather_data = get_weather(city, api_key)
    save_to_csv(weather_data)


if __name__ == "__main__":
    main()
