import json
import math
import time
import sys
import os
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Try importing Flask for the web server
try:
    from flask import Flask, request, render_template_string
except ImportError:
    Flask = None

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

class SearchEngine:
    def __init__(self, index_path="inverted_index.txt", 
                 lookup_path="lookup.json", 
                 docids_path="doc_ids.json",
                 pagerank_path="pagerank.json",
                 norms_path="doc_norms.json"):
        
        print("Loading search engine...")
        t0 = time.time()
        
        # Load Lookup Table
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(f"{lookup_path} not found! Run indexer.py first.")

        with open(lookup_path, "r", encoding="utf-8") as f:
            self.lookup = json.load(f)
            
        # Load Document ID Map
        with open(docids_path, "r", encoding="utf-8") as f:
            self.doc_mapper = json.load(f)
            self.doc_map = {int(k): v for k, v in self.doc_mapper.items()}
        
        # Load PageRank Scores
        try:
            with open(pagerank_path, "r", encoding="utf-8") as f:
                pr_data = json.load(f)
                self.pagerank = {int(k): v for k, v in pr_data.items()}
            print(f"Loaded PageRank for {len(self.pagerank)} docs.")
        except FileNotFoundError:
            print("Warning: pagerank.json not found. Ranking will rely on TF-IDF/Cosine.")
            self.pagerank = {}

        # Load Document Norms (for Cosine Similarity)
        try:
            with open(norms_path, "r", encoding="utf-8") as f:
                norms_data = json.load(f)
                self.doc_norms = {int(k): v for k, v in norms_data.items()}
        except FileNotFoundError:
            print("Warning: doc_norms.json not found. Cosine similarity will not be normalized.")
            self.doc_norms = {}

        self.index_path = index_path
        self.index_file = open(self.index_path, "r", encoding="utf-8")
        
        print(f"Ready in {time.time()-t0:.2f}s.")

    def __del__(self):
        if hasattr(self, 'index_file'):
            self.index_file.close()

    def preprocess_query(self, query: str):
        # Tokenize and stem
        tokens = tokenizer.tokenize(query.lower())
        stems = [stemmer.stem(tok) for tok in tokens]
        return stems

    def search(self, query, top_k=20):
        query_terms = self.preprocess_query(query)
        if not query_terms:
            return []

        termfreq = Counter(query_terms)
        scores = defaultdict(float)

        # For soft conjunction
        match_counts = defaultdict(int)

        # Query norm accumulator
        query_norm_sq = 0.0

        for term, fq in termfreq.items():
            offset = self.lookup.get(term)
            if offset is None:
                continue

            self.index_file.seek(offset)
            line = self.index_file.readline()
            if not line:
                continue

            try:
                idf, postings = json.loads(line)
            except:
                continue

            # MUST match index formula:
            # w_q = (1 + log(tf_q)) * idf
            w_q = (1.0 + math.log(fq)) * idf
            query_norm_sq += w_q * w_q

            for doc_id, w_d in postings:   # w_d is already TF-IDF from index
                scores[doc_id] += w_q * w_d     # dot product
                match_counts[doc_id] += 1

        query_norm = math.sqrt(query_norm_sq)
        results = []

        for doc_id, dot in scores.items():
            d_norm = self.doc_norms.get(doc_id, 0.0)

            if d_norm > 0 and query_norm > 0:
                cosine = dot / (query_norm * d_norm)
            else:
                cosine = 0.0

            # Soft conjunction boost (your previous logic)
            soft = match_counts[doc_id] * 0.25

            # PageRank
            pr = self.pagerank.get(doc_id, 0.0) * len(self.doc_norms) * 10.0

            final_score = cosine + soft + pr
            results.append((doc_id, final_score))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]


def run_web(port=8080):
    if Flask is None:
        print("Error: Flask is not installed. Please run: pip install flask")
        return

    try:
        engine = SearchEngine()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    app = Flask(__name__)
    
    HTML = """
    <html><head>
      <title>Search</title>
      <style>
        body { font-family: Trebuchet MS, sans-serif; margin: 30px; }
        input { width: 65%; padding: 8px; }
        button { padding: 8px 12px; }
        .meta { color:#666; margin-top: 10px; }
        li { margin: 10px 0; }
        .score { color:#c7d2fe; font-size: 0.9em; }
      </style>
    </head><body>
      <h2>Search</h2>
      <form>
        <input name="q" value="{{q|e}}" autofocus />
        <button>Search</button>
      </form>
      {% if q %}<div class="meta">{{count}} results • {{ms}} ms</div>{% endif %}
      {% if results %}
        <ol>
        {% for url, score in results %}
          <li><a href="{{url}}" target="_blank">{{url}}</a>
          <div class="score">score={{"%.4f"|format(score)}}</div></li>
        {% endfor %}
        </ol>
      {% endif %}
    </body></html>
    """

    @app.get("/")
    def home():
        q = request.args.get("q", "")
        results = []
        ms = 0.0
        
        if q.strip():
            t0 = time.perf_counter()
            hits = engine.search(q, top_k=20)
            ms = (time.perf_counter() - t0) * 1000
            
            results = []
            for doc_id, score in hits:
                url = engine.doc_map.get(doc_id, "<unknown>")
                results.append((url, score))
                
        return render_template_string(HTML, q=q, results=results, ms=f"{ms:.1f}", count=len(results))

    print(f"Starting web server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

def main():
    if "--web" in sys.argv:
        run_web()
    else:
        try:
            engine = SearchEngine()
        except Exception as e:
            print(f"Error: {e}")
            return

        print("\nConsole Search (Pass --web to run web server). Type 'quit' to exit.\n")
        while True:
            try:
                query = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not query: continue
            if query.lower() in {"quit", "exit", "q"}: break

            t0 = time.time()
            results = engine.search(query, top_k=20)
            elapsed = (time.time() - t0) * 1000

            print(f"\nFound {len(results)} results in {elapsed:.1f} ms\n")
            for rank, (doc_id, score) in enumerate(results, 1):
                url = engine.doc_map.get(doc_id, f"Doc {doc_id}")
                print(f"{rank:2d}. {url} ({score:.4f})")
            print()

if __name__ == "__main__":
    main()