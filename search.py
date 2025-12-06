import json
import math
import time
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

class SearchEngine:
    def __init__(self, index_path="inverted_index.txt", lookup_path="lookup.json", docids_path="doc_ids.json"):
        print("Loading search engine...")
        t0 = time.time()
        
        # Load the lightweight lookup table (Term -> Byte Offset)
        with open(lookup_path, "r", encoding="utf-8") as f:
            self.lookup = json.load(f)
            
        # Load document ID mapping
        with open(docids_path, "r", encoding="utf-8") as f:
            self.doc_mapper = json.load(f)
            # Ensure keys are ints
            self.doc_map = {int(k): v for k, v in self.doc_mapper.items()}
            
        self.index_path = index_path
        # We keep the file handle open for queries
        self.index_file = open(self.index_path, "r", encoding="utf-8")
        
        print(f"Loaded {len(self.lookup)} terms and {len(self.doc_map)} docs in {time.time()-t0:.2f}s.")

    def __del__(self):
        if hasattr(self, 'index_file'):
            self.index_file.close()

    def preprocess_query(self, query: str):
        tokens = tokenizer.tokenize(query.lower())
        stems = [stemmer.stem(tok) for tok in tokens]
        return stems

    def search(self, query, top_k=20):
        query_terms = self.preprocess_query(query)
        if not query_terms:
            return []
        
        termfreq = Counter(query_terms)
        scores = defaultdict(float)
        
        # We process each query term
        for term, fq in termfreq.items():
            # 1. Get offset from lookup
            offset = self.lookup.get(term)
            if offset is None:
                continue
            
            # 2. Seek to position in file
            self.index_file.seek(offset)
            
            # 3. Read and parse the line
            line = self.index_file.readline()
            if not line:
                continue
                
            # Line format: JSON string [idf, [[doc_id, score], ...]]
            try:
                data = json.loads(line)
                idf = data[0]
                postings = data[1]
            except json.JSONDecodeError:
                continue

            # 4. Calculate query weight (TF-IDF for query)
            w_q = (1.0 + math.log(fq)) * idf
            
            # 5. Accumulate scores
            # Note: doc_info is [doc_id, tfidf_score]
            for doc_id, doc_score in postings:
                scores[doc_id] += w_q * doc_score

        # Sort and return top K
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

def main():
    try:
        engine = SearchEngine()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run indexer.py first.")
        return

    print("\nOptimized Console Search. Type 'quit' to exit.\n")

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
        results = engine.search(query, top_k=20)
        elapsed_ms = (time.time() - t0) * 1000

        print(f"\nFound {len(results)} results in {elapsed_ms:.1f} ms\n")

        if not results:
            print("No results.\n")
            continue

        for rank, (doc_id, score) in enumerate(results, start=1):
            url = engine.doc_map.get(doc_id, f"<unknown doc {doc_id}>")
            print(f"{rank:2d}. {url}")
            print(f"score = {score:.4f}")
        print()

if __name__ == "__main__":
    main()