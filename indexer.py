import json
import os
import requests
from urllib.parse import urlparse
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import psutil

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

MAX_MEMORY_MB = 750

def current_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

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
    # {term: tf}
    freq = defaultdict(float)
    token_count = 0

    # Processes token and their weight
    def process(text, weight):
        nonlocal token_count
        if not text: 
            return
        text_lower = text.lower()
        for token in tokenizer.tokenize(text_lower):
            token_count += 1
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

    return freq, token_count


def create_partial_index(part_index, run_number, output_dir):
    path = os.path.join(output_dir, f"partial_idx_{run_number}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(part_index, f)

    part_index.clear()
    return path

def merge_partial_indexes(partial_idx_paths, final_output):
    final_index = defaultdict(dict)

    for path in partial_idx_paths:
        with open(path, "r", encoding="utf-8") as f:
            idx = json.load(f)

        for term, postings in idx.items():
            for doc_id, tf in postings.items():
                final_index[term][doc_id] = tf

    inverted = {}
    for term, posting_dict in final_index.items():
        postings_list = [
            {"doc_id": int(doc_id), "tf": tf}
            for doc_id, tf in posting_dict.items()
        ]
        inverted[term] = {
            "df": len(postings_list),
            "postings": postings_list
        }

    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(inverted, f, indent=2)

    return inverted

def get_inverted_index(root_folder, output_path="inverted_index.json", docids_output="doc_ids.json", partial_idxs_folder="partial_indexes"):
    # {doc_id: tf}
    os.makedirs(partial_idxs_folder, exist_ok=True)
    doc_map = {}
    doc_count = 0
    run = 0
    partials = []
    mem_idx = defaultdict(dict)

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

            tokens, total_tokens = extract_tokens(html)

            for term, tf in tokens.items():
                mem_idx[term][doc_id] = tf

            if current_memory_mb() >= MAX_MEMORY_MB:
                partial_path = create_partial_index(mem_idx, run, partial_idxs_folder)
                partials.append(partial_path)

                mem_idx = defaultdict(dict)
                run += 1

            doc_id += 1

            """
            for token, count in tokens.items():
                term_posting[token][doc_id] = count


            """

    if mem_idx:
        partial_path = create_partial_index(mem_idx, run, partial_idxs_folder)
        partials.append(partial_path)

    with open(docids_output, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, indent=2)

    final_index = merge_partial_indexes(partials, output_path)

    return final_index, doc_map

def main():
    index, doc_map = get_inverted_index("DEV/")
    return index

if __name__ == "__main__":
    main()