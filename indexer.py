import json
import os
import math
import sys
import shutil
import heapq
import hashlib
from urllib.parse import urlparse, urljoin
from contextlib import ExitStack
from collections import defaultdict
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

BLOCK_SIZE = 2000
WEIGHTS = {
    "title": 3.0, 
    "h1": 2.5, 
    "h2": 2.0, 
    "h3": 1.5, 
    "bold": 1.3, 
    "normal": 1.0
}

class SimHash:
    def __init__(self, threshold=1):
        self.seen_fingerprints = set()
        self.threshold = threshold  # max Hamming distance for near-duplicates

    def _hash_func(self, token):
        # Returns a stable 64-bit hash
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def compute(self, token_weights):
        """
        Computes 64-bit SimHash fingerprint for a dictionary of {token: weight}.
        """
        v = [0] * 64
        for token, weight in token_weights.items():
            h = self._hash_func(token)
            for i in range(64):
                if h & (1 << i):
                    v[i] += weight
                else:
                    v[i] -= weight
        
        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def _hamming_distance(self, a, b):
        return (a ^ b).bit_count()

    def is_duplicate(self, fingerprint):
        """
        Returns True if a near-duplicate (Hamming distance <= threshold)
        has been seen before.
        """
        for fp in self.seen_fingerprints:
            if self._hamming_distance(fp, fingerprint) <= self.threshold:
                return True

        self.seen_fingerprints.add(fingerprint)
        return False

def compute_pagerank(adjacency_list, doc_map_rev, num_docs, iterations=20, d=0.85):
    print(f"Computing PageRank for {num_docs} documents...")
    link_structure = defaultdict(list)
    for source_url, targets in adjacency_list.items():
        if source_url not in doc_map_rev: continue
        source_id = doc_map_rev[source_url]
        for t_url in targets:
            if t_url in doc_map_rev and t_url != source_url:
                target_id = doc_map_rev[t_url]
                link_structure[source_id].append(target_id)

    pr = {i: 1.0 / num_docs for i in range(num_docs)}
    for _ in range(iterations):
        new_pr = {i: 0.0 for i in range(num_docs)}
        sink_pr = 0
        for i in range(num_docs):
            if i not in link_structure or not link_structure[i]:
                sink_pr += pr[i]
        
        base_mass = (1.0 - d) / num_docs
        sink_mass = (d * sink_pr) / num_docs
        total_added = base_mass + sink_mass

        for source_id, targets in link_structure.items():
            share = pr[source_id] / len(targets)
            for target_id in targets:
                new_pr[target_id] += share
        
        for i in range(num_docs):
            new_pr[i] = (new_pr[i] * d) + total_added
        pr = new_pr
    return pr

def check_if_xml(input_content):
    content_start = input_content.lstrip()[:100].lower()
    return content_start.startswith("<?xml") or ("<" in content_start and "/>" in content_start)

def parse_html(html):
    if check_if_xml(html): return BeautifulSoup(html, features="xml")
    return BeautifulSoup(html, "html.parser")

def extract_content(soup):
    for tag in soup(["script", "style", "noscript"]): tag.decompose()
    freq = defaultdict(float)
    def process(text, weight):
        if not text: return
        for token in tokenizer.tokenize(text.lower()):
            stem = stemmer.stem(token)
            freq[stem] += weight

    if soup.title and soup.title.string: process(soup.title.string, WEIGHTS["title"])
    for tag_name, w in [("h1", WEIGHTS["h1"]), ("h2", WEIGHTS["h2"]), ("h3", WEIGHTS["h3"])]:
        for tag in soup.find_all(tag_name): process(tag.get_text(separator=" "), w)
    for tag in soup.find_all(["b", "strong"]): process(tag.get_text(separator=" "), WEIGHTS["bold"])
    skip_tags = {"title", "h1", "h2", "h3", "b", "strong", "script", "style", "noscript"}
    for item in soup.find_all(string=True):
        if item.parent and item.parent.name and item.parent.name.lower() not in skip_tags:
            process(item, WEIGHTS["normal"])
    return freq

def extract_links(soup, base_url):
    links = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        try:
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            clean_url = parsed.scheme + "://" + parsed.netloc + parsed.path
            if parsed.query: clean_url += "?" + parsed.query
            if clean_url.startswith("http"): links.add(clean_url)
        except: continue
    return list(links)

def write_partial_index(partial_index, block_num, output_dir="temp_indices"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"part_{block_num}.txt")
    sorted_terms = sorted(partial_index.keys())
    with open(filename, "w", encoding="utf-8") as f:
        for term in sorted_terms:
            postings_str = ";".join([f"{did}:{tf:.2f}" for did, tf in partial_index[term]])
            f.write(f"{term}|{postings_str}\n")
    return filename

def merge_indices(temp_files, output_path, lookup_path, N):
    print(f"Merging {len(temp_files)} partial indices...")
    with ExitStack() as stack:
        files = [stack.enter_context(open(fn, "r", encoding="utf-8")) for fn in temp_files]
        heap = []
        for i, f in enumerate(files):
            line = f.readline()
            if line:
                term, content = line.strip().split("|", 1)
                heapq.heappush(heap, (term, i, content))
        
        final_index = open(output_path, "w", encoding="utf-8")
        lookup = {}
        
        # Dictionary to store the sum of squares of weights for each doc
        # used for cosine similarity normalization
        doc_norms = defaultdict(float)

        current_term = None
        current_postings = []
        
        while heap:
            term, file_idx, content = heapq.heappop(heap)
            if current_term is not None and term != current_term:
                df = len(current_postings)
                idf = math.log(N / df) if df > 0 else 0
                final_postings = []
                for doc_id, tf in current_postings:
                    # TF-IDF Weight Calculation
                    w_d = (1 + math.log(tf))
                    score = w_d * idf
                    
                    # Accumulate sum of squares for vector magnitude
                    doc_norms[doc_id] += score ** 2
                    
                    final_postings.append([doc_id, score])
                
                offset = final_index.tell()
                lookup[current_term] = offset
                final_index.write(json.dumps([idf, final_postings]) + "\n")
                current_postings = []

            current_term = term
            for item in content.split(";"):
                d_str, tf_str = item.split(":")
                current_postings.append((int(d_str), float(tf_str)))
            
            next_line = files[file_idx].readline()
            if next_line:
                nt, nc = next_line.strip().split("|", 1)
                heapq.heappush(heap, (nt, file_idx, nc))
        
        # Handle last term
        if current_term is not None and current_postings:
            df = len(current_postings)
            idf = math.log(N / df) if df > 0 else 0
            final_postings = []
            for doc_id, tf in current_postings:
                w_d = (1 + math.log(tf))
                score = w_d * idf
                doc_norms[doc_id] += score ** 2
                final_postings.append([doc_id, score])
            
            offset = final_index.tell()
            lookup[current_term] = offset
            final_index.write(json.dumps([idf, final_postings]) + "\n")

        final_index.close()
        
        # Save lookup
        with open(lookup_path, "w", encoding="utf-8") as f: 
            json.dump(lookup, f)
            
        # Compute Sqrt and save Doc Norms
        final_norms = {k: math.sqrt(v) for k, v in doc_norms.items()}
        with open("doc_norms.json", "w", encoding="utf-8") as f:
            json.dump(final_norms, f)
            
    print("Merge complete.")

def get_inverted_index(root_folder):
    doc_map = {}
    doc_map_rev = {}
    adjacency_list = {}
    partial_index = defaultdict(list)
    temp_files = []
    simhasher = SimHash()
    doc_count = 0
    duplicates = 0
    block_num = 0
    
    print("Indexing documents...")
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.endswith(".json"): continue
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f: data = json.load(f)
            except: continue
            
            url = data.get("url", file_path)
            html = data.get("content", "")
            
            soup = parse_html(html)
            tokens = extract_content(soup)
            if not tokens: continue
            
            fingerprint = simhasher.compute(tokens)
            if simhasher.is_duplicate(fingerprint):
                duplicates += 1
                continue
            
            doc_id = doc_count
            doc_count += 1
            doc_map[doc_id] = url
            doc_map_rev[url] = doc_id
            adjacency_list[url] = extract_links(soup, url)

            for token, tf in tokens.items(): partial_index[token].append((doc_id, tf))
            
            if doc_count % BLOCK_SIZE == 0:
                print(f"  Processed {doc_count} docs. (Duplicates skipped: {duplicates})")
                temp_files.append(write_partial_index(partial_index, block_num))
                partial_index.clear()
                block_num += 1

    if partial_index:
        temp_files.append(write_partial_index(partial_index, block_num))
    
    print(f"Total Unique Docs: {doc_count}. Duplicates removed: {duplicates}")
    with open("doc_ids.json", "w", encoding="utf-8") as f: json.dump(doc_map, f, indent=2)
    
    pagerank_scores = compute_pagerank(adjacency_list, doc_map_rev, doc_count)
    with open("pagerank.json", "w", encoding="utf-8") as f: json.dump(pagerank_scores, f, indent=2)

    if temp_files:
        merge_indices(temp_files, "inverted_index.txt", "lookup.json", doc_count)
        shutil.rmtree("temp_indices", ignore_errors=True)
    else: print("No documents found.")

def main():
    root_folder = sys.argv[1] if len(sys.argv) >= 2 else "DEV/"
    get_inverted_index(root_folder)

if __name__ == "__main__":
    main()