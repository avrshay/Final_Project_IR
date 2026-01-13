import math
import pickle
from flask import Flask, request, jsonify
import os
import hashlib
import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
import pickle

import sys
from inverted_index_gcp import InvertedIndex

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", "may", "first", "see", "history", "people", "one", "two", "part", "thumb", "including", "second", "following", "many", "however", "would", "became"]
# Define Stopwords
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)




class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

#Global:
def load_pickle_or_default(filename, default_val=None):
    if default_val is None:
        default_val = {}

    if not os.path.exists(filename):
        print(f"WARNING: File '{filename}' not found! Using empty default.")
        return default_val

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            print(f"Successfully loaded {filename}")
            return data
    except Exception as e:
        print(f"ERROR: Failed to load '{filename}'. Reason: {e}")
        return default_val


BASE_PATH = "/home/moriyc"
INDEX_PATH = os.path.join(BASE_PATH, "index.pkl")
DL_PATH = os.path.join(BASE_PATH, "DL_dict.pkl")
PR_PATH = os.path.join(BASE_PATH, "pr_dict.pkl")
ID2TITLE_PATH = os.path.join(BASE_PATH, "ID2TITLE_dict.pkl")

try:
    inverted = InvertedIndex.read_index(BASE_PATH, 'index')
    print("Inverted Index loaded successfully.")
except Exception as e:
    print(f"CRITICAL WARNING: Could not load Inverted Index. Search will not work. Reason: {e}")
    inverted = None

DL = load_pickle_or_default(DL_PATH)
pagerank = load_pickle_or_default(PR_PATH)
id_to_title = load_pickle_or_default(ID2TITLE_PATH)





class BM25_from_index:
    """
    BM25 wrapper that reads posting lists from disk only when needed.
    """

    def __init__(self, index, DL, base_dir=".", k1=1.5, b=0.7):
        """
        :param index: InvertedIndex object (after pickle load)
        :param DL: dict {doc_id: doc_length}
        :param base_dir: folder containing the .bin posting files
        """
        self.index = index
        self.DL = DL
        self.N = len(DL)
        total_len = 0
        for l in DL.values():
            total_len += l
        self.AVGDL = total_len / self.N
        self.k1 = k1
        self.b = b
        self.base_dir = base_dir

    def calc_idf(self, tokens):
        idf = {}
        for t in tokens:
            df = self.index.df.get(t, 0)
            idf[t] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        return idf

    def search(self, query, N=10):
        query_tokens = [tok.group() for tok in RE_WORD.finditer(query.lower())]
        query_tokens = [t for t in query_tokens if t not in all_stopwords]
        idf = self.calc_idf(query_tokens)
        scores = Counter()

        for term in query_tokens:
            if term not in self.index.df:
                continue

            # Read posting list for this term from disk (lazy load)
            posting_list = self.index.read_a_posting_list(self.base_dir, term)

            for doc_id, freq in posting_list:
                doc_len = self.DL.get(doc_id)
                if doc_len is None:
                    continue
                num = idf[term] * freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.AVGDL))
                scores[doc_id] += num / den
        return scores.most_common(N)

    def search_with_pagerank(self, query, N=10, alpha=0.8, M=30):
        bm25_results = self.search(query, N=M)
        reranked = []
        for doc_id, bm25_score in bm25_results:
            pr_score = pagerank.get(doc_id, 0.0)
            final_score = alpha * bm25_score + (1 - alpha) * pr_score
            reranked.append((doc_id, final_score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:N]


#  Create the search engine
bm25 = BM25_from_index(inverted, DL,base_dir="")

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    search_results = bm25.search_with_pagerank(query, 10, 0.8, 20)

    predicted_ids = [str(doc_id) for doc_id, _ in search_results]
    res = [(doc_id, id_to_title.get(str(doc_id), "UNKNOWN TITLE")) for doc_id in predicted_ids]


    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for doc_id in wiki_ids:
        doc_id = int(doc_id)
        pr_score = pagerank.get(doc_id, 0.0)
        res.append(pr_score)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
