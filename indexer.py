import json
import os
import math
import sys
import shutil
import heapq
from contextlib import ExitStack
from collections import defaultdict
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
stemmer = PorterStemmer()

# Configuration
BLOCK_SIZE = 2000  # Number of documents to process before dumping to disk
WEIGHTS = {
    "title": 3.0, 
    "h1": 2.5, 
    "h2": 2.0, 
    "h3": 1.5, 
    "bold": 1.3, 
    "normal": 1.0
}

def check_if_xml(input_content):
    content_start = input_content.lstrip()[:100].lower()
    return content_start.startswith("<?xml") or ("<" in content_start and "/>" in content_start)

def extract_tokens(html):
    if check_if_xml(html):
        soup = BeautifulSoup(html, features="xml")
    else:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    freq = defaultdict(float)

    def process(text, weight):
        if not text: return
        for token in tokenizer.tokenize(text.lower()):
            stem = stemmer.stem(token)
            freq[stem] += weight

    if soup.title and soup.title.string:
        process(soup.title.string, WEIGHTS["title"])

    for tag_name, w in [("h1", WEIGHTS["h1"]), ("h2", WEIGHTS["h2"]), ("h3", WEIGHTS["h3"])]:
        for tag in soup.find_all(tag_name):
            process(tag.get_text(separator=" "), w)

    for tag in soup.find_all(["b", "strong"]):
        process(tag.get_text(separator=" "), WEIGHTS["bold"])

    skip_tags = {"title", "h1", "h2", "h3", "b", "strong", "script", "style", "noscript"}
    for item in soup.find_all(string=True):
        if item.parent and item.parent.name and item.parent.name.lower() not in skip_tags:
            process(item, WEIGHTS["normal"])

    return freq

def write_partial_index(partial_index, block_num, output_dir="temp_indices"):
    """
    Writes a partial index to disk.
    Format: term|doc_id:tf;doc_id:tf;...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"part_{block_num}.txt")
    
    # Sort terms to allow for efficient merging later
    sorted_terms = sorted(partial_index.keys())
    
    with open(filename, "w", encoding="utf-8") as f:
        for term in sorted_terms:
            postings = partial_index[term] # list of (doc_id, tf)
            # Serialize postings to string
            postings_str = ";".join([f"{did}:{tf:.2f}" for did, tf in postings])
            f.write(f"{term}|{postings_str}\n")
            
    return filename

def merge_indices(temp_files, output_path, lookup_path, N):
    """
    Merges multiple sorted partial index files into one final index.
    Calculates TF-IDF on the fly.
    """
    print(f"Merging {len(temp_files)} partial indices...")
    
    # Open all temp files
    with ExitStack() as stack:
        files = [stack.enter_context(open(fn, "r", encoding="utf-8")) for fn in temp_files]
        
        # Priority queue for merging: (term, file_index, line_content)
        heap = []
        
        # Initialize heap with first line from each file
        for i, f in enumerate(files):
            line = f.readline()
            if line:
                term, content = line.strip().split("|", 1)
                heapq.heappush(heap, (term, i, content))
        
        final_index = open(output_path, "w", encoding="utf-8")
        lookup = {}
        
        current_term = None
        current_postings = [] # List of (doc_id, tf)
        
        while heap:
            term, file_idx, content = heapq.heappop(heap)
            
            # If we see a new term, process the previous accumulated term
            if current_term is not None and term != current_term:
                # 1. Calculate stats
                df = len(current_postings)
                idf = math.log(N / df) if df > 0 else 0
                
                # 2. Build final postings with TF-IDF scores
                final_postings = []
                for doc_id, tf in current_postings:
                    # Log-frequency weighting
                    w_d = (1 + math.log(tf))
                    score = w_d * idf
                    final_postings.append([doc_id, score])
                
                # 3. Write to file
                # Save current file position for lookup
                offset = final_index.tell()
                lookup[current_term] = offset
                
                # Format: JSON list for easy parsing: [idf, [[doc_id, score], ...]]
                # We store IDF in the file so searcher doesn't need to recalculate it
                data = json.dumps([idf, final_postings])
                final_index.write(data + "\n")
                
                # Reset for new term
                current_postings = []

            current_term = term
            
            # Parse the content from the temp file (doc_id:tf;doc_id:tf)
            for item in content.split(";"):
                doc_id_str, tf_str = item.split(":")
                current_postings.append((int(doc_id_str), float(tf_str)))
            
            # Read next line from the file that supplied the current term
            next_line = files[file_idx].readline()
            if next_line:
                next_term, next_content = next_line.strip().split("|", 1)
                heapq.heappush(heap, (next_term, file_idx, next_content))
        
        # Process the very last term
        if current_term is not None and current_postings:
            df = len(current_postings)
            idf = math.log(N / df) if df > 0 else 0
            final_postings = []
            for doc_id, tf in current_postings:
                w_d = (1 + math.log(tf))
                score = w_d * idf
                final_postings.append([doc_id, score])
            
            offset = final_index.tell()
            lookup[current_term] = offset
            data = json.dumps([idf, final_postings])
            final_index.write(data + "\n")

        final_index.close()
        
        # Save lookup table
        with open(lookup_path, "w", encoding="utf-8") as f:
            json.dump(lookup, f)
            
    print("Merge complete.")

def get_inverted_index(root_folder):
    doc_map = {}
    doc_count = 0
    
    partial_index = defaultdict(list)
    temp_files = []
    block_num = 0
    
    print("Indexing documents...")
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(dirpath, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                continue

            url = data.get("url", file_path)
            html = data.get("content", "")

            doc_id = doc_count
            doc_map[doc_id] = url
            doc_count += 1

            # Extract tokens
            tokens = extract_tokens(html)

            # Add to partial index
            for token, tf in tokens.items():
                partial_index[token].append((doc_id, tf))
            
            # Check if block size exceeded
            if doc_count % BLOCK_SIZE == 0:
                print(f"  Processed {doc_count} docs. Dumping block {block_num}...")
                temp_file = write_partial_index(partial_index, block_num)
                temp_files.append(temp_file)
                partial_index.clear() # Free memory
                block_num += 1

    # Dump remaining documents
    if partial_index:
        print(f"  Dumping final block {block_num}...")
        temp_file = write_partial_index(partial_index, block_num)
        temp_files.append(temp_file)
        partial_index.clear()

    # Save doc_ids map
    with open("doc_ids.json", "w", encoding="utf-8") as f:
        json.dump(doc_map, f, indent=2)

    # Merge all partial indices
    if temp_files:
        merge_indices(temp_files, "inverted_index.txt", "lookup.json", doc_count)
        
        # Cleanup temp files
        shutil.rmtree("temp_indices", ignore_errors=True)
    else:
        print("No documents found.")

    # Report
    index_size = os.path.getsize("inverted_index.txt") / (1024 * 1024)
    lookup_size = os.path.getsize("lookup.json") / (1024 * 1024)
    print("\nReport:")
    print(f"Total Documents: {doc_count}")
    print(f"Index File Size: {index_size:.2f} MB")
    print(f"Lookup File Size: {lookup_size:.2f} MB")

def main():
    root_folder = sys.argv[1] if len(sys.argv) >= 2 else "DEV/"
    get_inverted_index(root_folder)

if __name__ == "__main__":
    main()