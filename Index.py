from nltk.corpus.reader.api import tempfile

corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)

NUM_BUCKETS = 124
def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

def word_count(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in
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
  '''
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  # YOUR CODE HERE
  # remove stop_words
  token_without_stopWords=[]
  for token in tokens:
    if token not in all_stopwords:
      token_without_stopWords.append(token)
  tf_tokens=Counter(token_without_stopWords)
  list_tokens=[]
  for token, tf in tf_tokens.items():
    list_tokens.append((token,(id,tf)))
  return list_tokens

def reduce_word_counts(unsorted_pl):
  ''' Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples
  Returns:
  --------
    list of tuples
      A sorted posting list.
  '''
  # YOUR CODE HERE
  return sorted(unsorted_pl,key=lambda x: x[0])

def partition_postings_and_write(postings):
  ''' A function that partitions the posting lists into buckets, writes out
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
  '''
   # YOUR CODE HERE
  token_bucket= postings.map(lambda x: (token2bucket_id(x[0]),x)) #Placing a suitable bucket for each token, along with its posting list
  union_bucket= token_bucket.groupByKey() #Token consolidation and posting list by bucket number
  return union_bucket.map(lambda x: InvertedIndex.write_a_posting_list(x)) #For each bucket we will return the list of locations in its file
# merge the posting locations into a single dict and run more tests (5 points)

def merge_posting_locs(locs_list):
    super_posting_locs = defaultdict(list)
    for posting_loc in locs_list:
        for k, v in posting_loc.items():
            super_posting_locs[k].extend(v)
    return super_posting_locs

#title

title_doc_text_pairs = parquetFile.limit(1000).select("text", "id").rdd # לשנות בהתאם
title_word_counts  = title_doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))
title_postings  = title_word_counts .groupByKey().mapValues(reduce_word_counts)
# filtering postings and calculate df
title_postings_filtered  = title_postings.filter(lambda x: len(x[1])>10)
title_w2df  = calculate_df(title_postings_filtered )
title_w2df_dict  = title_w2df .collectAsMap()
#  partitioning for the different buckets
title_locs_list = partition_postings_and_write(sc.parallelize(title_postings_filtered)).collect()
#merge
title_posting_locs = merge_posting_locs(title_locs_list)
# Create inverted index instance
title_index = InvertedIndex()
# Adding the posting locations dictionary to the inverted index
title_index.posting_locs = title_posting_locs
# Add the token - df dictionary to the inverted index
title_index.df = w2df_dict
# write the global stats out
title_index.write_index('./title_index', 'index')

#body

body_doc_text_pairs = parquetFile.limit(1000).select("text", "id").rdd # לשנות בהתאם
body_word_counts   = body_doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))
body_postings   = body_word_counts  .groupByKey().mapValues(reduce_word_counts)
# filtering postings and calculate df
body_postings_filtered   = body_postings.filter(lambda x: len(x[1])>10)
body_w2df   = calculate_df(body_postings_filtered  )
body_w2df_dict   = body_w2df  .collectAsMap()
#  partitioning for the different buckets
body_locs_list = partition_postings_and_write(sc.parallelize(body_postings_filtered)).collect()
#merge
body_posting_locs = merge_posting_locs(body_locs_list)
# Create inverted index instance
body_index = InvertedIndex()
# Adding the posting locations dictionary to the inverted index
body_index.posting_locs = body_posting_locs
# Add the token - df dictionary to the inverted index
body_index.df = w2df_dict
# write the global stats out
body_index.write_index('./body_indices', 'index')

#anchor

anchor_doc_text_pairs = parquetFile.limit(1000).select("text", "id").rdd # לשנות בהתאם
anchor_word_counts = anchor_doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))
anchor_postings  = anchor_word_counts.groupByKey().mapValues(reduce_word_counts)
# filtering postings and calculate df
anchor_postings_filtered    = anchor_postings.filter(lambda x: len(x[1])>10)
anchor_w2df  = calculate_df(anchor_postings_filtered)
anchor_w2df_dict = anchor_w2df.collectAsMap()
# partitioning for the different buckets
anchor_locs_list = partition_postings_and_write(sc.parallelize(anchor_postings)).collect()
#merge
anchor_posting_locs = merge_posting_locs(anchor_locs_list)
# Create inverted index instance
anchor_index = InvertedIndex()
# Adding the posting locations dictionary to the inverted index
anchor_index.posting_locs = anchor_posting_locs
# Add the token - df dictionary to the inverted index
anchor_index.df = w2df_dict
# write the global stats out
anchor_index.write_index('./anchor_index', 'index')
