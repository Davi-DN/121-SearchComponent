import json
import math
import time
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

class SearchEngine:
    def __init__(self, index_path="inverted_index.txt", 
                 lookup_path="lookup.json", 
                 docids_path="doc_ids.json",
                 pagerank_path="pagerank.json"):
        
        print("Loading search engine...")
        t0 = time.time()
        
        with open(lookup_path, "r", encoding="utf-8") as f:
            self.lookup = json.load(f)
            
        with open(docids_path, "r", encoding="utf-8") as f:
            self.doc_mapper = json.load(f)
            self.doc_map = {int(k): v for k, v in self.doc_mapper.items()}
        
        # Load PageRank scores
        try:
            with open(pagerank_path, "r", encoding="utf-8") as f:
                pr_data = json.load(f)
                # Ensure keys are ints and values are floats
                self.pagerank = {int(k): v for k, v in pr_data.items()}
            print(f"Loaded PageRank scores for {len(self.pagerank)} documents.")
        except FileNotFoundError:
            print("Warning: pagerank.json not found. Ranking will rely solely on TF-IDF.")
            self.pagerank = {}

        self.index_path = index_path
        self.index_file = open(self.index_path, "r", encoding="utf-8")
        
        print(f"Ready in {time.time()-t0:.2f}s.")

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
        
        doc_norm_sq = defaultdict(float)
        query_terms_norm_sq = 0.0

        # 1. Calculate TF-IDF Relevance Scores
        for term, fq in termfreq.items():
            offset = self.lookup.get(term)
            if offset is None:
                continue
            
            self.index_file.seek(offset)
            line = self.index_file.readline()
            if not line: continue
                
            try:
                data = json.loads(line)
                idf = data[0]
                postings = data[1]
            except json.JSONDecodeError:
                continue

            w_q = (1.0 + math.log(fq)) * idf
            query_terms_norm_sq += w_q * w_q
            
            for doc_id, doc_score in postings:
                w_d = doc_score
                scores[doc_id] += w_q * w_d
                doc_norm_sq[doc_id] += w_d * w_d

        if not scores: 
            return []

        q_norm = math.sqrt(query_terms_norm_sq) or 1.0
        cosine_scores = {}

        for doc_id, numerator in scores.items():
            d_norm = math.sqrt(doc_norm_sq[doc_id]) or 1.0
            cosine_scores[doc_id] = numerator / (q_norm * d_norm)

        
        # 2. Combine with PageRank
        # We need to normalize or weight PageRank so it doesn't dominate or disappear.
        # Simple approach: weighted addition.
        # Adjust PR_WEIGHT based on how strong you want the "authority" signal to be.
        PR_WEIGHT = 10.0 
        N = len(self.doc_map)
        
        final_results = []
        for doc_id, cos_score in cosine_scores.items():
            pr_score = self.pagerank.get(doc_id, 0.0)
            
            # Hybrid Score
            # Use log(pr) if pr values vary wildly, or just raw pr if they are normalized.
            # PageRank sum is 1, so individual values are tiny (~1/N).
            # We scale it up or it will be negligible compared to TF-IDF (which is usually > 1.0).
            # Let's scale PR by N (number of docs) to make it roughly 1.0 on average.
            
            scaled_pr = pr_score * N 
            
            final_score = cos_score + (PR_WEIGHT * scaled_pr)
            
            final_results.append((doc_id, final_score))

        # Sort
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:top_k]
        

def main():
    try:
        engine = SearchEngine()
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    print("\nSearch Engine with PageRank & Deduplication. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query: continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        t0 = time.time()
        results = engine.search(query, top_k=20)
        elapsed_ms = (time.time() - t0) * 1000

        print(f"\nFound {len(results)} results in {elapsed_ms:.1f} ms\n")

        for rank, (doc_id, score) in enumerate(results, start=1):
            url = engine.doc_map.get(doc_id, f"<unknown doc {doc_id}>")
            print(f"{rank:2d}. {url}")
            print(f"   (Score: {score:.4f})")
        print()

if __name__ == "__main__":
    main()
