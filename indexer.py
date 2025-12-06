import json
import os
import requests
from urllib.parse import urlparse
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import sys

from bs4 import MarkupResemblesLocatorWarning
import warnings

import hashlib
import argparse
import heapq

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# "all alphanumeric sequences"
tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

# Weight of types of text: certain texts have higher weight than others
WEIGHTS = {
    "title": 3, 
    "h1": 3, 
    "h2": 2, 
    "h3": 2, 
    "bold": 2, 
    "normal": 1
 }

SKIP = {"title","h1","h2","h3","b","strong","script","style","noscript"}

# exact duplicate
def check_duplicate(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

# strips whitespace and removes fragments
def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if "#" in u:
        u = u.split("#", 1)[0]
    return u

# check xml
def check_if_xml(input):
    content_start = input.lstrip()[:100].lower()
    if content_start.startswith("<?xml") or "<" in content_start and "/>" in content_start:
        return True
    else: 
        return False


def stem_tokens(text: str):
    return [stemmer.stem(t) for t in tokenizer.tokenize((text or "").lower())]

# extract text with weighting
def extract_tokens(html: str):
    """
    Returns:
      tf_map: term -> weighted_tf (int)
      doc_len: token count of visible text (int)
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()

    tf = defaultdict(int)

    def process(text: str, w: int):
        if not text:
            return
        for tok in tokenizer.tokenize(text.lower()):
            tf[stemmer.stem(tok)] += w
    if soup.title and soup.title.string:
        process(soup.title.string, WEIGHTS["title"])

    for tag_name, w in [("h1", WEIGHTS["h1"]), ("h2", WEIGHTS["h2"]), ("h3", WEIGHTS["h3"])]:
        for tag in soup.find_all(tag_name):
            process(tag.get_text(" ", strip=True), w)

    for tag in soup.find_all(["b", "strong"]):
        process(tag.get_text(" ", strip=True), WEIGHTS["bold"])

    for item in soup.find_all(string=True):
        parent = item.parent.name.lower() if item.parent and item.parent.name else ""
        if parent in SKIP:
            continue
        process(str(item), WEIGHTS["normal"])

    visible = soup.get_text(" ", strip=True)
    doc_len = len(tokenizer.tokenize(visible.lower()))
    if doc_len <= 0:
        doc_len = 1
    return tf, doc_len


#  writes a partial inverted index to disk.
def flush_partial(partial: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(partial.keys()):
            posting_map = partial[term]
            postings = [[doc_id, tf] for doc_id, tf in posting_map.items()]
            postings.sort(key=lambda pair: pair[1], reverse=True)
            obj = {"df": len(postings), "postings": postings}
            f.write(term + "\t" + json.dumps(obj) + "\n")

# merges sorted partial files
def merge(partial_files, out_index_bin, out_lexicon_json, max_postings_store=50000):
    # OPEN ALL partials (list!)
    fps = [open(p, "r", encoding="utf-8") for p in partial_files]

    def read_item(i):
        line = fps[i].readline()
        if not line:
            return None
        term, js = line.rstrip("\n").split("\t", 1)
        return term, js

    heap = []
    for i in range(len(fps)):
        it = read_item(i)
        if it:
            heap.append((it[0], i, it[1]))
    heapq.heapify(heap)

    lex = {}
    with open(out_index_bin, "wb") as out:
        while heap:
            term, i, js = heapq.heappop(heap)
            merged = json.loads(js)["postings"]
            # merge with same term
            while heap and heap[0][0] == term:
                _, j, js2 = heapq.heappop(heap)
                merged.extend(json.loads(js2)["postings"])
                nxtj = read_item(j)
                if nxtj:
                    heapq.heappush(heap, (nxtj[0], j, nxtj[1]))
            nxti = read_item(i)
            if nxti:
                heapq.heappush(heap, (nxti[0], i, nxti[1]))

            full_df = len(merged)
            merged.sort(key=lambda pair: pair[1], reverse=True)
            stored = merged[:max_postings_store]

            obj = {"df": full_df, "postings": stored}
            line = (term + "\t" + json.dumps(obj) + "\n").encode("utf-8")

            offset = out.tell()
            out.write(line)
            lex[term] = [offset, len(line), full_df]

    for f in fps:
        f.close()

    with open(out_lexicon_json, "w", encoding="utf-8") as f:
        json.dump(lex, f)

def build(root_folder: str, out_dir: str, flush_every_docs: int, max_postings_store: int):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "partials"), exist_ok=True)
    partial = defaultdict(lambda: defaultdict(int))
    partial_files = []
    doc_ids = {}
    doc_len = {}
    seen_fp = set()
    doc_id = 0
    flush_id = 0

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue

            path = os.path.join(dirpath, filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            url = normalize_url(data.get("url", path))
            html = data.get("content", "")

            # check duplicate fingerprint
            fp = check_duplicate(html)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)

            tf_map, dl = extract_tokens(html)
            doc_ids[doc_id] = url
            doc_len[doc_id] = dl

            for term, tf in tf_map.items():
                partial[term][doc_id] = int(tf)

            doc_id += 1

            if doc_id % flush_every_docs == 0:
                p = os.path.join(out_dir, "partials", f"partial_{flush_id:03d}.txt")
                flush_partial(partial, p)
                partial_files.append(p)
                partial.clear()
                flush_id += 1
                print(f"[flush] {p}")

    if partial:
        p = os.path.join(out_dir, "partials", f"partial_{flush_id:03d}.txt")
        flush_partial(partial, p)
        partial_files.append(p)
        partial.clear()
        flush_id += 1
        print(f"[flush] {p}")

    if flush_id < 3:
        print("fewer than 3 partial flushes. Must lower --flush every doc")

    # write doc maps
    with open(os.path.join(out_dir, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in doc_ids.items()}, f)
    with open(os.path.join(out_dir, "doc_len.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in doc_len.items()}, f)

    # merge partials
    out_index = os.path.join(out_dir, "index.bin")
    out_lex = os.path.join(out_dir, "lexicon.json")
    merge(partial_files, out_index, out_lex, max_postings_store=max_postings_store)

    print(f"[done] docs={len(doc_ids)}")
    print(f"[done] index={out_index}")
    print(f"[done] lexicon={out_lex}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder of .json pages with keys {url, content}")
    ap.add_argument("--out", required=True, help="output index folder")
    ap.add_argument("--flush_every_docs", type=int, default=3000, help="flush partial index every N docs (ensure >=3 flushes)")
    ap.add_argument("--max_postings_store", type=int, default=50000, help="store only top M postings per term (keeps search fast)")
    args = ap.parse_args()
    build(args.root, args.out, args.flush_every_docs, args.max_postings_store)

# python indexer.py --root DEV --out index_out --flush_every_docs 3000
if __name__ == "__main__":
    main()