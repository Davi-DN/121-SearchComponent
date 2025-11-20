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

    # becomes a python dict inverted_index with df and postings
    with open(index_path, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)

    # loads object with string keys
    with open(docids_path, "r", encoding="utf-8") as f:
        doc_mapper = json.load(f)

    # convert json string keys to ints
    doc_map = {int(k): v for k, v in doc_mapper.items()}
    N = len(doc_map)
    print(f"Loaded {len(inverted_index)} terms and {N} documents.")
    return inverted_index, doc_map, N

def preprocess_query(query: str):
    # all the stemmed tokens in a list 
    tokens = tokenizer.tokenize(query.lower())
    stems = [stemmer.stem(tok) for tok in tokens]
    return stems

def search(query, inverted_index, doc_map, N, top_k=20):
    # list of stems
    query_terms = preprocess_query(query)
    if not query_terms:
        return []
    
    # dict of term counter
    termfreq = Counter(query_terms)
    # stores total tf-idf style score for that doc
    scores = defaultdict(float)
    for term, fq in termfreq.items():
        if term not in inverted_index:
            continue

        entry = inverted_index[term]
        df = entry["df"] # number of documents with the term
        postings = entry["postings"] # list of doc_id, tf, etc.

        if df == 0:
            continue

        # log(N/df) -> less frequent terms get bigger idf
        # rare terms have more impact like "simhash" compared to "uci"
        idf = math.log(N / df)

        # query term weight
        # multiply how often it appears *
        w_q = (1.0 + math.log(fq)) * idf

        # loop over all with the term
        for posting in postings:
            doc_id = posting["doc_id"]
            tf_doc = posting["tf"]

            # safety reasons, shouldn't go lower than 0.
            if tf_doc <= 0:
                continue

            # weight of document term
            # term-frequency * idf for classic weighting
            w_d = (1.0 + math.log(tf_doc)) * idf
            scores[doc_id] += w_q * w_d

    # sort docs in descending order
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # return top_k list of (doc_id, score) which is 20 right now
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

        # timing the search
        t0 = time.time()
        results = search(query, inverted_index, doc_map, N, top_k=20)
        elapsed_ms = (time.time() - t0) * 1000

        print(f"\nFound {len(results)} results in {elapsed_ms:.1f} ms\n")

        if not results:
            print("No results.\n")
            continue

        # prints results
        for rank, (doc_id, score) in enumerate(results, start=1):
            url = doc_map.get(doc_id, f"<unknown doc {doc_id}>")
            print(f"{rank:2d}. {url}")
            print(f"score = {score:.4f}")
        print()


if __name__ == "__main__":
    main()