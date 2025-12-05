import json
import math
import time
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

def load_index(index_path="inverted_index.json", docids_path="doc_ids.json"):
    print("Loading index into memory (simple version)...")
    with open(index_path, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)

    with open(docids_path, "r", encoding="utf-8") as f:
        raw_doc_map = json.load(f)

    # convert json keys to ints
    doc_map = {int(k): v for k, v in raw_doc_map.items()}
    N = len(doc_map)
    print(f"Loaded {len(inverted_index)} terms and {N} documents.")
    return inverted_index, doc_map, N

def preprocess_query(query: str):
    tokens = tokenizer.tokenize(query.lower())
    stems = [stemmer.stem(tok) for tok in tokens]
    return stems

def search(query, inverted_index, doc_map, N, top_k=20):

    query_terms = preprocess_query(query)
    if not query_terms:
        return []
    
    termfreq = Counter(query_terms)

    scores = defaultdict(float)
    for term, fq in termfreq.items():
        if term not in inverted_index:
            continue

        entry = inverted_index[term]
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

def main():
    inverted_index, doc_map, N = load_index()

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
        results = search(query, inverted_index, doc_map, N, top_k=5)
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

# python3 C:\Users\dngo0\OneDrive\Documents\CS121\HW3\SearchComponent\121-SearchComponent\search.py tom