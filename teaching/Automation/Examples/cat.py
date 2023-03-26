import requests
import csv
import time


def get_cat_fact():
    url = "https://catfact.ninja/fact"
    response = requests.get(url)
    fact_data = response.json()
    return fact_data["fact"]


def save_to_csv(facts, filename="cat_facts.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["fact"])
        for fact in facts:
            writer.writerow([fact])


def main():
    cat_facts = []
    while len(cat_facts) < 100:
        try:
            fact = get_cat_fact()
            cat_facts.append(fact)
            print(f"Fact {len(cat_facts)}: {fact}")
            if len(cat_facts) < 100:
                time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    save_to_csv(cat_facts)


if __name__ == "__main__":
    main()
