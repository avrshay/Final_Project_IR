# preprocessing.py
from collections import Counter
from config import *
from inverted_index_gcp import *

# ==========================================
# 4. Core Logic Functions
# ==========================================

def partition_postings_and_write(postings):
    ''' Partitions posting lists into buckets and writes to disk. '''
    # Bucket assignment
    buckets = postings.map(lambda x: (token2bucket_id(x[0]), x))

    # Group By Key (term) - Note: In Spark logic this usually groups by bucket_id first if mapped that way,
    # but here we need to group by bucket to write together.
    # The logic in the original code: map -> (bucket, (term, pl)) -> groupByKey -> (bucket, iter[(term, pl)])
    buckets = buckets.groupByKey()

    # Write to the disk
    # returns RDD of posting locations
    buckets = buckets.map(InvertedIndex.write_a_posting_list)
    return buckets

def word_count(text, id):
    ''' Count the frequency of each word in `text` (tf) that is not included in
    `all_stopwords` and return entries that will go into our posting lists.
    '''
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]

    #Stopword
    filtered_tokens = [w for w in tokens if w not in all_stopwords]

    #Stemming
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    after_stem = []
    for t in filtered_tokens:
        after_stem.append(stemmer.stem(t))

    # Create a map {term : counter}
    map_counts = Counter(after_stem)

    # Create the required tuples for each term in map
    result = [(term, (id, count)) for term, count in map_counts.items()]
    return result

def reduce_word_counts(unsorted_pl):
    ''' Returns a sorted posting list by wiki_id. '''
    return sorted(unsorted_pl)

def calculate_df(postings):
    ''' Takes a posting list RDD and calculate the df for each token. '''
    return postings.mapValues(len)