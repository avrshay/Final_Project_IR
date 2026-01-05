# BM25.py
import math
import builtins
from nltk.stem.porter import PorterStemmer
from config import *
from inverted_index_colab import *
from collections import Counter

"""#### BM25
As a reminder:

To use BM25 we will need to following parameters:

* $k1$ and $b$ - free parameters
* $f(t_i,D)$ - term frequency of term $t_i$ in document $D$
* |$D$|- is the length of the document $D$ in terms
* $avgdl$ -  average document length
* $IDF$ - which is calculted as follows: $ln(\frac{(N-n(t_i)+0.5}{n(t_i)+0.5)}+1)$, where $N$ is the total number of documents in the collection, and $n(t_i)$ is the number of documents containing $t_i$ (e.g., document frequency (df)).

"""

class BM25_from_index:
    """
    Best Match 25.
    """

    def __init__(self, index, DL, k1=1.5, b=0.75):
        """
        :param index: Inverted index object
        :param DL: Dictionary representing document length {doc_id: length}
        :param k1: float, default 1.5
        :param b: float, default 0.75
        """
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(DL)
        self.AVGDL = builtins.sum(DL.values()) / self.N
        self.stemmer = PorterStemmer()

    def calc_idf(self, list_of_tokens):
        """
        Calculates IDF for terms in the query.
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                idf[term] = 0
        return idf

    @staticmethod
    def read_posting_list(inverted, w):
        ''' Reads a posting list from disk for a specific word w. '''
        with closing(MultiFileReader()) as reader:
            locs = inverted.posting_locs[w] #!!!The posting list files (.bin files) need to be in the current directory.

            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def search(self, query, N=10):
        """
        Rank documents for a given query.
        """
        # 1. Preprocessing
        query_tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        query_filtered_tokens = [w for w in query_tokens if w not in all_stopwords]

        query_after_stem = []
        for t in query_filtered_tokens:
            query_after_stem.append(self.stemmer.stem(t))

        # 2. Calculate IDF
        self.idf = self.calc_idf(query_after_stem)

        # 3. Score Candidates (Optimized - Term-at-a-Time)
        scores = Counter() # Accumulates the scores for each document

        for term in query_after_stem:

            if term in self.index.df:

                posting_list = self.read_posting_list(self.index, term)

                idf_val = self.idf[term]
                for doc_id, freq in posting_list:

                    doc_len = self.DL.get(doc_id)

                    if doc_len is None:
                        continue

                    numerator = idf_val * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.AVGDL))


                    scores[doc_id] += (numerator / denominator)

        # 4. Sort and Return
        return scores.most_common(N)
