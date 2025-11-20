import json
import os
import requests
from urllib.parse import urlparse
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

# Weight of types of text: certain texts have higher weight than others
WEIGHTS = {
    "title": 3.0,
    "h1": 2.5,
    "h2": 2.0,
    "h3": 1.5,
    "bold": 1.3,
    "normal": 1.0
}

def check_if_xml(input):
    content_start = input.lstrip()[:100].lower()
    if content_start.startswith("<?xml") or "<" in content_start and "/>" in content_start:
        return True
    else: 
        return False

def extract_tokens(html):
    # Checks if input is an XML: if so, parses it using an XML parser, if not, then uses HTML parser
    if check_if_xml(html):
        soup = BeautifulSoup(html, features="xml")
    else:
        soup = BeautifulSoup(html, "html.parser")

    # not useful for indexing
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # if token doesn't exist, starts at 0.0
    freq = defaultdict(float)

    # Processes token and their weight
    def process(text, weight):
        if not text: 
            return
        for token in tokenizer.tokenize(text.lower()):
            stem = stemmer.stem(token)
            freq[stem] += weight

    # Title
    if soup.title and soup.title.string:
        process(soup.title.string, WEIGHTS["title"])

    # Headings
    for tag_name, w in [("h1", WEIGHTS["h1"]),
                             ("h2", WEIGHTS["h2"]),
                             ("h3", WEIGHTS["h3"])]:
        for tag in soup.find_all(tag_name):
            process(tag.get_text(separator=" "), w)

    # Bold text
    for tag in soup.find_all(["b", "strong"]):
        process(tag.get_text(separator=" "), WEIGHTS["bold"])

    # Normal text (excluding special tags)
    skip_tags = {"title", "h1", "h2", "h3", "b", "strong", "script", "style", "noscript"}
    for item in soup.find_all(string=True):
        if item.parent and item.parent.name:
            parent_name = item.parent.name.lower()
        else:
            parent_name = ""
        if parent_name in skip_tags:
            continue
        process(item, WEIGHTS["normal"])

    return freq


def get_inverted_index(root_folder, output_path="inverted_index.json", docids_output="doc_ids.json"):
    # {doc_id: tf}
    term_posting = defaultdict(dict)
    doc_map = {}
    doc_count = 0

    for dirpath, _ , filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(dirpath, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            url = data.get("url", file_path)
            html = data.get("content", "")

            doc_id = doc_count
            doc_count += 1
            doc_map[doc_id] = url

            tokens = extract_tokens(html)

            for token, count in tokens.items():
                term_posting[token][doc_id] = count

    # index structure
    inverted_index = {}
    for term, posting_dict in term_posting.items():
        postings_list = [
            {"doc_id": doc_id, "tf": tf}
            for doc_id, tf in posting_dict.items()
        ]
        inverted_index[term] = {
            "df": len(postings_list),
            "postings": postings_list
        }


    # writing to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=2)

    with open(docids_output, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, indent=2)

    num_docs = len(doc_map)
    num_unique_tokens = len(inverted_index)
    index_size_kb = os.path.getsize(output_path) / 1024.0


    # Our report
    print("Report:")
    print(f"Indexed documents: {num_docs}")
    print(f"Unique tokens: {num_unique_tokens}")
    print(f"Index size on disk: {index_size_kb:.2f} KB")
    return inverted_index, doc_map

def main():
    index, doc_map = get_inverted_index("DEV/")
    return index

if __name__ == "__main__":
    main()