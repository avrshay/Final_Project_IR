import os
import sys
import signal
import pickle
from collections import defaultdict
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from graphframes import GraphFrame

# Custom imports
import config
import preprocessing
from pagerank import *
from inverted_index_gcp import InvertedIndex

# ==========================================
# 1. Environment Setup
# ==========================================

GF_PKG = "graphframes:graphframes:0.8.2-spark3.1-s_2.12"

spark = SparkSession.builder \
    .appName("IR-Project-GCP") \
    .config("spark.jars.packages", GF_PKG) \
    .config("spark.sql.shuffle.partitions", "16") \
    .config("spark.default.parallelism", "16") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# ==========================================
# 2. Data Configuration & Loading
# ==========================================

project_id = 'ir-final-project-2025'
data_bucket_name = '319134458_214906935'

checkpoint_dir = f"gs://{data_bucket_name}/checkpoints"
sc.setCheckpointDir(checkpoint_dir)
print(f"Checkpoint directory set to: {checkpoint_dir}")

print(f"Spark initialized: {spark.version}")

path = f"gs://{data_bucket_name}/*.parquet"

print(f"Reading data from: {path}")
try:

    parquetFile = spark.read.parquet(path)
    print("Successfully connected to GCS and read parquet files.")
except Exception as e:
    print(f"Error reading from GCS: {e}")
    sys.exit(1)

# ==========================================
# 4. Inverted Index Pipeline
# ==========================================

print("Starting Inverted Index creation...")
doc_text_pairs = parquetFile.select("id", "text").where(col("text").isNotNull()).rdd

# 2. Calculate Word Counts
word_counts = doc_text_pairs.flatMap(lambda x: preprocessing.word_count(x[0], x[1])).cache()

# 3. Create Posting Lists
postings = word_counts.groupByKey().mapValues(preprocessing.reduce_word_counts)

# 4. Filter Low Frequency Terms
postings_filtered = postings.filter(lambda x: len(x[1]) > 10)

# 5. Calculate Document Frequency (DF)
w2df_rdd = preprocessing.calculate_df(postings_filtered)
w2df_dict = w2df_rdd.collectAsMap()

# 6. Calculate Document Length (DL)
DL = word_counts \
    .map(lambda x: (x[1][0], x[1][1])) \
    .reduceByKey(lambda a, b: a + b) \
    .collectAsMap()

print("Saving DL.pkl...")
with open('DL.pkl', 'wb') as f:
    pickle.dump(DL, f)

# 7. Save ID to Title Map
print("Saving id_to_title.pkl...")
id_to_title_map = parquetFile.select("id", "title").where(col("title").isNotNull()).rdd.collectAsMap()
with open('id_to_title.pkl', 'wb') as f:
    pickle.dump(id_to_title_map, f)

# 8. Write Index Partitioned to Disk
print("Writing Index to Local Disk...")
posting_locs_rdd = preprocessing.partition_postings_and_write(postings_filtered)
posting_locs_list = posting_locs_rdd.collect()

# 9. Merge & Finalize Index Object
super_posting_locs = defaultdict(list)
for posting_loc in posting_locs_list:
    for k, v in posting_loc.items():
        super_posting_locs[k].extend(v)

inverted = InvertedIndex()
inverted.posting_locs = super_posting_locs
inverted.df = w2df_dict
inverted.write_index('.', 'index')

# ==========================================
# 3. PageRank Calculation
# ==========================================

pages_links = parquetFile.select("id", "anchor_text").where(col("anchor_text").isNotNull()).rdd

print(f"pages_links created")

try:
    create_page_rank(data_bucket_name, pages_links)
    print(f"PageRank ready")
except Exception as e:
    print(f"ERROR in PageRank: {e}")
    # Continue anyway - to build the inverted index
    sys.exit(1)

# ==========================================
# 5. Upload Results to GCS
# ==========================================
print("Uploading files to GCS...")

# Upload to Bucket
os.system(f"gsutil cp index.pkl gs://{data_bucket_name}/")
os.system(f"gsutil cp DL.pkl gs://{data_bucket_name}/")
os.system(f"gsutil cp id_to_title.pkl gs://{data_bucket_name}/")

# Only upload pagerank if it exists
if os.path.exists('pagerank.pkl'):
    os.system(f"gsutil cp pagerank.pkl gs://{data_bucket_name}/")

os.system(f"gsutil -m cp *.bin gs://{data_bucket_name}/")

print(f"DONE! All index files uploaded to gs://{data_bucket_name}/")
