import json
import math
import time
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import hashlib
import heapq
import argparse
import os

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

from flask import Flask, request, render_template_string

def current_ms() -> float:
    return time.time() * 1000.0

# page rank
def pagerank(outlinks, N, d=0.85, iters=40):
    pr = [1.0 / N] * N
    outdeg = [0] * N
    for u, vs in outlinks.items():
        outdeg[u] = len(vs)

    for _ in range(iters):
        new = [(1-d) / N] * N
        for u, vs in outlinks.items():
            if not vs: 
                continue
            share = d * pr[u] / len(vs)
            for v in vs:
                new[v] += share
        pr = new
    return pr

# query preprocessing: lowercase, tokenize alphanumerics and stem
def preprocess_text(text: str):
    toks = tokenizer.tokenize((text or "").lower())
    return [stemmer.stem(t) for t in toks]

# load json file
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_url(u: str) -> str:
    # doc_ids map
    return (u or "").strip()

def read_term_line(fbin, offset, length):
    fbin.seek(offset)
    line = fbin.read(length).decode("utf-8")
    _term, js = line.rstrip("\n").split("\t", 1)
    obj = json.loads(js)
    return int(obj["df"]), obj["postings"]

# Computes tf-idf ranking:
def search(query, lex, doc_len, N, index_bin_path, top_k=20, champion_limit=2000):
    q_terms = preprocess_text(query)
    if not q_terms:
        return []

    tfq = Counter(q_terms)
    scores = defaultdict(float)

    with open(index_bin_path, "rb") as fbin:
        for term, qtf in tfq.items():
            meta = lex.get(term)
            if not meta:
                continue
            offset, length, df = meta
            df = int(df)
            if df <= 0:
                continue

            # tf-idf
            idf = math.log((N + 1) / (df + 1)) + 1.0
            wq = (1.0 + math.log(qtf)) * idf

            _df2, postings = read_term_line(fbin, int(offset), int(length))

            for doc_id, tf in postings[:champion_limit]:
                tf = int(tf)
                if tf <= 0:
                    continue
                wd = (1.0 + math.log(tf)) * idf
                scores[int(doc_id)] += wq * wd

    if not scores:
        return []
    for d in list(scores.keys()):
        dl = doc_len.get(d, 1)
        scores[d] /= math.sqrt(max(1, dl))

    # good for top-k retrieval without sorting all docs
    return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])

# the client interface: loads small metadata (lexicon + doc maps), then loops queries.
def console(index_dir, champion_limit):
    lex = load_json(os.path.join(index_dir, "lexicon.json"))
    doc_ids = {int(k): v for k, v in load_json(os.path.join(index_dir, "doc_ids.json")).items()}
    doc_len = {int(k): int(v) for k, v in load_json(os.path.join(index_dir, "doc_len.json")).items()}
    N = len(doc_ids)

    index_bin = os.path.join(index_dir, "index.bin")

    print("\nConsole search. Type 'quit' to exit.\n")
    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q.lower() in {"quit","exit","q"}:
            break

        t0 = time.perf_counter()
        hits = search(q, lex, doc_len, N, index_bin, top_k=20, champion_limit=champion_limit)
        ms = (time.perf_counter() - t0) * 1000

        print(f"\nFound {len(hits)} results in {ms:.1f} ms\n")
        for i, (doc_id, score) in enumerate(hits, 1):
            print(f"{i:2d}. {doc_ids.get(doc_id,'<unknown>')}")
            print(f"score={score:.4f}")
        print()

# using flask for gui
def run_web(index_dir, port, champion_limit):
    lex = load_json(os.path.join(index_dir, "lexicon.json"))
    doc_ids = {int(k): v for k, v in load_json(os.path.join(index_dir, "doc_ids.json")).items()}
    doc_len = {int(k): int(v) for k, v in load_json(os.path.join(index_dir, "doc_len.json")).items()}
    N = len(doc_ids)
    index_bin = os.path.join(index_dir, "index.bin")

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
        {% for url,score in results %}
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
            hits = search(q, lex, doc_len, N, index_bin, top_k=20, champion_limit=champion_limit)
            ms = (time.perf_counter() - t0) * 1000
            results = [(doc_ids.get(d, "<unknown>"), s) for d, s in hits]
        return render_template_string(HTML, q=q, results=results, ms=f"{ms:.1f}", count=len(results))

    app.run(host="0.0.0.0", port=port, debug=True)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("console")
    c1.add_argument("--index_dir", required=True)
    c1.add_argument("--champion", type=int, default=2000)

    c2 = sub.add_parser("web")
    c2.add_argument("--index_dir", required=True)
    c2.add_argument("--port", type=int, default=5000)
    c2.add_argument("--champion", type=int, default=2000)

    args = ap.parse_args()

    # python search.py console --index_dir index_out
    # python search.py web --index inverted_index.json --anchors anchor_index.json --docids doc_ids.json --port 5000
    if args.cmd == "console":
        console(args.index_dir, args.champion)
    else:
        run_web(args.index_dir, args.port, args.champion)

if __name__ == "__main__":
    main()
