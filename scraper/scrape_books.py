import requests
from bs4 import BeautifulSoup
import json
import time
from tqdm import tqdm

BASE_URL = "https://acharyaprashant.org"
URL = "https://acharyaprashant.org/en/books/all"

headers = {
    "User-Agent": "Mozilla/5.0"
}


def get_book_links():

    response = requests.get(URL, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")

    cards = soup.select('a[href^="/en/books/"]')

    links = []

    for card in cards:

        href = card.get("href")

        links.append(BASE_URL + href)

    return list(set(links))


headers = {
    "User-Agent": "Mozilla/5.0"
}

def scrape_book(url):

    res = requests.get(url, headers=headers)
    res.encoding = "utf-8"

    soup = BeautifulSoup(res.text, "lxml")

    # title
    title_tag = soup.select_one("h1")
    title = title_tag.text.strip() if title_tag else ""

    # description
    desc_tag = soup.select_one("span.leading-normal.dynamicHTMLContainer")
    description = desc_tag.get_text(" ", strip=True) if desc_tag else ""

    chapters = []

    scripts = soup.select("script[data-sveltekit-fetched]")

    for script in scripts:

        if not script.string:
            continue

        try:
            outer = json.loads(script.string)

            body_str = outer.get("body")
            if not body_str:
                continue

            body = json.loads(body_str)

            content = body.get("content")
            if not content:
                continue

            enum_mask = content.get("enumMask", {}).get("value", {})
            sub_contents = enum_mask.get("subContents", {})

            # ebook chapters
            if "1" in sub_contents:
                chapter_data = sub_contents["1"]["value"].get("chapters", [])

                chapters = [c["title"].strip() for c in chapter_data if "title" in c]

                break

        except Exception as e:
            continue

    return {
        "title": title,
        "url": url,
        "description": description,
        "chapters": chapters
    }
    
def main():

    links = get_book_links()

    print("Total books:", len(links))

    books = []

    for link in tqdm(links[:2]):

        try:

            book = scrape_book(link)

            books.append(book)

            time.sleep(0.5)

        except Exception as e:

            print("Error:", link)

    with open("books2.json", "w", encoding="utf-8") as f:

        json.dump(books, f, ensure_ascii=False, indent=2)

    print("Saved books2.json")


if __name__ == "__main__":
    main()