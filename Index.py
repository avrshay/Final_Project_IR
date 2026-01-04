from collections import *
from nltk.corpus.reader.api import tempfile
import inverted_index_colab
from Tokenazier import *
import hashlib
from collections import Counter
from graphframes import GraphFrame
from pyspark.sql import SparkSession

GF_PKG = "graphframes:graphframes:0.8.3-spark3.5-s_2.12"

# Stop any old session quietly
try:
    spark.stop()
except:
    pass

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("IR-GraphFrames")
    .config("spark.jars.packages", GF_PKG)   # <-- brings in the JAR
    .config("spark.ui.port", "0")            # free UI port (avoid 'connection refused')
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)


sc = spark.sparkContext


NUM_BUCKETS = 124


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()
def word_count(text, id):
    """ Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  """
    tokens = tokenize(text)

    # remove stop_words
    token_without_stopWords = remove_stopWord(tokens)
    token_stemming = stemming(tokens)
    tf_tokens = Counter(token_stemming)
    list_tokens = []
    for token, tf in tf_tokens.items():
        list_tokens.append((token, (id, tf)))
    return list_tokens


def reduce_word_counts(unsorted_pl):
    """ Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples
  Returns:
  --------
    list of tuples
      A sorted posting list.
  """
    return sorted(unsorted_pl, key=lambda x: x[0])

def calculate_df(postings):
  ''' Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  '''
    # YOUR CODE HERE
  df_tokens=postings.mapValues(lambda x:len(x))
  return df_tokens


def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


def partition_postings_and_write(postings):
    """ A function that partitions the posting lists into buckets, writes out
  all posting lists in a bucket to disk, and returns the posting locations for
  each bucket. Partitioning should be done through the use of `token2bucket`
  above. Writing to disk should use the function  `write_a_posting_list`, a
  static method implemented in inverted_index_colab.py under the InvertedIndex
  class.
  Parameters:
  -----------
    postings: RDD
      An RDD where each item is a (w, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each item is a posting locations dictionary for a bucket. The
      posting locations maintain a list for each word of file locations and
      offsets its posting list was written to. See `write_a_posting_list` for
      more details.
  """
    # YOUR CODE HERE
    token_bucket = postings.map(
        lambda x: (token2bucket_id(x[0]), x))  # Placing a suitable bucket for each token, along with its posting list
    union_bucket = token_bucket.groupByKey()  # Token consolidation and posting list by bucket number
    return union_bucket.map(lambda x: inverted_index_colab.write_a_posting_list(
        x))  # For each bucket we will return the list of locations in its file


# merge the posting locations into a single dict and run more tests (5 points)

def merge_posting_locs(locs_list):
    super_posting_locs = defaultdict(list)
    for posting_loc in locs_list:
        for k, v in posting_loc.items():
            super_posting_locs[k].extend(v)
    return super_posting_locs


# title
def create_inverted_index(pages, type):
    doc_text_pairs = pages.limit(1000).select(type, "id").rdd  # לשנות בהתאם
    word_counts = doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))
    postings = word_counts.groupByKey().mapValues(reduce_word_counts)
    # filtering postings and calculate df
    postings_filtered = postings.filter(lambda x: len(x[1]) > 10)
    w2df = calculate_df(postings_filtered)
    w2df_dict = w2df.collectAsMap()
    #  partitioning for the different buckets
    locs_list = partition_postings_and_write(sc.parallelize(postings_filtered)).collect()
    # merge
    posting_locs = merge_posting_locs(locs_list)
    # Create inverted index instance
    index = inverted_index_colab.InvertedIndex()
    # Adding the posting locations dictionary to the inverted index
    index.posting_locs = posting_locs
    # Add the token - df dictionary to the inverted index
    index.df = w2df_dict
    # write the global stats out
    index.write_index(f'./{type}_index', 'index')
    return index

