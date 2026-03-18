import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

df = pd.read_csv("books_links.csv")

books_data = []

for i, row in df.iterrows():

    url = row["link"]

    try:

        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        title = soup.find("h1").text.strip()

        description = soup.find("p").text.strip()

        chapters = []

        for li in soup.find_all("li"):
            chapters.append(li.text.strip())

        books_data.append({
            "title": title,
            "description": description,
            "chapters": " | ".join(chapters)
        })

        print(title)

        time.sleep(1)

    except:
        pass

pd.DataFrame(books_data).to_csv("books_full.csv", index=False)