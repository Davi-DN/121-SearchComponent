import json
import math
import time
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import os
import ijson
import decimal

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

"""
def convert_decimal(obj):
    if isinstance(obj, dict):
        return {k: convert_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal(v) for v in obj]
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    else:
        return obj


def load_index(index_path="inverted_index.json", docids_path="doc_ids.json", index_jsonl="inverted_index.jsonl", idx_to_idx_file="idx_to_idx.json"):
    idx_to_idx = {}
    offset = 0

    with open(index_path, "r", encoding="utf-8") as idx_in, open(index_jsonl, "w", encoding="utf-8") as idx_out:
        parser = ijson.kvitems(idx_in, "")
        for term, entry in parser:
            entry = convert_decimal(entry)    


            line = json.dumps({term: entry}) + "\n"
            idx_out.write(line)
            idx_to_idx[term] = offset
            offset += len(line.encode("utf-8"))

    with open(idx_to_idx_file, "w", encoding="utf-8") as f:
        json.dump(idx_to_idx, f)
"""

def preprocess_query(query: str):
    tokens = tokenizer.tokenize(query.lower())
    stems = [stemmer.stem(tok) for tok in tokens]
    return stems


def get_docs(term, index_jsonl="inverted_index.jsonl", idx_to_idx_file="idx_to_idx.json"):
    with open(idx_to_idx_file, "r", encoding="utf-8") as f:
        idx_to_idx = json.load(f)
        
    if term not in idx_to_idx:
        return None
    
    offset = idx_to_idx[term]
    with open(index_jsonl, "rb") as f:
        f.seek(offset)
        line = f.readline()
        if not line:
            return None
        
        data = json.loads(line.decode("utf-8"))
        return data.get(term)  # {"df":..., "postings":[...]}

def search(query, doc_map, N, top_k=20):
    query_terms = preprocess_query(query)
    if not query_terms:
        return []
    
    termfreq = Counter(query_terms)
    scores = defaultdict(float)

    for term, fq in termfreq.items():
        entry = get_docs(term)
        if not entry:
            continue

        df = entry["df"]
        postings = entry["postings"]

        if df == 0:
            continue

        # idf(t) = log(N/df)
        idf = math.log(N / df)

        # query term weight
        w_q = (1.0 + math.log(fq)) * idf

        # loop over all with the term
        for posting in postings:
            doc_id = posting["doc_id"]
            tf_doc = posting["tf"]

            if tf_doc <= 0:
                continue

            # weight of document term
            w_d = (1.0 + math.log(tf_doc)) * idf
            scores[doc_id] += w_q * w_d

    # sort docs
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # return top_k list of (doc_id, score)
    return sorted_docs[:top_k]

def load_doc_map(docids_path="doc_ids.json"):
    with open(docids_path, "r", encoding="utf-8") as f:
        raw_doc_map = json.load(f)
    doc_map = {int(k): v for k, v in raw_doc_map.items()}
    N = len(doc_map)
    return doc_map, N

def main():
    doc_map, N = load_doc_map()

    print("\nSimple console search. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        t0 = time.time()
        results = search(query, doc_map, N, top_k=5)
        elapsed_ms = (time.time() - t0) * 1000
        print(f"\nFound {len(results)} results in {elapsed_ms:.1f} ms\n")

        if not results:
            print("No results.\n")
            continue
        for rank, (doc_id, score) in enumerate(results, start=1):
            url = doc_map.get(doc_id, f"<unknown doc {doc_id}>")
            print(f"{rank:2d}. {url}")
            print(f"score = {score:.4f}")
        print()


if __name__ == "__main__":
    main()
