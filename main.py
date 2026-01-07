import os
import sys
import signal
import pickle
from collections import defaultdict

# PySpark imports
from pyspark.sql import SparkSession

# Custom imports (assuming these files exist in the same directory)
import config
import preprocessing
from inverted_index_gcp import InvertedIndex

# ==========================================
# 1. Environment Setup
# ==========================================
# Note: Dependencies like 'graphframes' should be installed in the environment
# before running this script (e.g., via pip install).

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
GF_PKG = "graphframes:graphframes:0.8.3-spark3.5-s_2.12"

# ==========================================
# 2. Spark Session Initialization
# ==========================================
# Stop any active session to avoid conflicts
try:
    from pyspark.context import SparkContext

    SparkContext.getOrCreate().stop()
except:
    pass

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("IR-Project-GCP")
    .config("spark.jars.packages", GF_PKG)
    .config("spark.ui.port", "0")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

sc = spark.sparkContext
print("Spark:", spark.version, "| GraphFrames Python wrapper imported OK")

# ==========================================
# 3. Data Configuration & Download
# ==========================================

# --- Authentication ---
try:
    from google.colab import auth

    AUTH_TIMEOUT = 100


    def handler(signum, frame):
        raise Exception("Authentication timeout!")


    signal.signal(signal.SIGALRM, handler)
    signal.alarm(AUTH_TIMEOUT)
    auth.authenticate_user()
    signal.alarm(0)

except ImportError:
    pass
except Exception as e:
    print(f"Auth Warning: {e}")
    signal.alarm(0)

# --- Configuration ---
project_id = 'ir-final-project-2025'
data_bucket_name = '319134458_214906935'

# Set project configuration
os.system(f"gcloud config set project {project_id}")

path = f"gs://{data_bucket_name}/*.parquet"

try:
    parquetFile = spark.read.parquet(path)
    print("Successfully connected to GCS and read parquet files.")
except Exception as e:
    print(f"Error reading from GCS: {e}")
    print("Make sure the .parquet files exist in the bucket path specified!")
    sys.exit(1)

"""
# Handle data paths and cleanup
#if "wikidata_preprocessed" in os.environ:
#    del os.environ["wikidata_preprocessed"]


# Create directory and download data if not exists
try:
   os.makedirs("wikidumps", exist_ok=True)

   if os.environ.get("wikidata_preprocessed") is None:
       print("Downloading data from bucket...")
       # Using os.system instead of magic command (!)
       os.system(f"gsutil -u {project_id} cp gs://{data_bucket_name}/multistream1_preprocessed.parquet wikidumps/")
except Exception as e:
   print(f"Download warning: {e}")


# Define the reading path
try:
    if os.environ.get("wikidata_preprocessed") is not None:
        path = os.environ["wikidata_preprocessed"] + "/wikidumps/*"
    else:
        path = "wikidumps/*"
except:
    path = "wikidumps/*"
"""

# ==========================================
# 4. Main Processing Pipeline
# ==========================================

# 1. Read Parquet file
print(f"Reading data from: {path}")
parquetFile = spark.read.parquet(path)

# 2. Limit to 1000 docs for testing (as per original code)
#doc_text_pairs = parquetFile.limit(1000).select("text", "id").rdd

doc_text_pairs = parquetFile.select("text", "id").rdd

# 3. Calculate Word Counts (Term Frequency)
# Using the function from preprocessing.py
word_counts = doc_text_pairs.flatMap(lambda x: preprocessing.word_count(x[0], x[1]))

# 4. Create Posting Lists
# GroupByKey -> (Term, List of (DocID, TF))
# MapValues -> Sort and reduce the list
postings = word_counts.groupByKey().mapValues(preprocessing.reduce_word_counts)

# 5. Filter Low Frequency Terms
postings_filtered = postings.filter(lambda x: len(x[1]) > 10)

# 6. Calculate Document Frequency (DF)
# Using the function from preprocessing.py
w2df_rdd = preprocessing.calculate_df(postings_filtered)
w2df_dict = w2df_rdd.collectAsMap()

# Calculate the length of each document (DL)
DL = word_counts \
    .map(lambda x: (x[1][0], x[1][1])) \
    .reduceByKey(lambda a, b: a + b) \
    .collectAsMap()

# Save DL to disk
print("Saving DL.pkl...")
with open('DL.pkl', 'wb') as f:
    pickle.dump(DL, f)

# Save the id+title map for each doc to disk
print("Saving id_to_title.pkl...")
id_to_title_map = parquetFile.limit(1000).select("id", "title").rdd.collectAsMap()
with open('id_to_title.pkl', 'wb') as f:
    pickle.dump(id_to_title_map, f)


# 7. Write Index to Disk
print("Writing Index to Locl Disk...")
# Using the function from preprocessing.py to partition and write
posting_locs_rdd = preprocessing.partition_postings_and_write(postings_filtered)
posting_locs_list = posting_locs_rdd.collect()


# 8. Merge & Finalize Index
super_posting_locs = defaultdict(list)
for posting_loc in posting_locs_list:
    for k, v in posting_loc.items():
        super_posting_locs[k].extend(v)

# Create Inverted Index instance and save locally
inverted = InvertedIndex()
inverted.posting_locs = super_posting_locs
inverted.df = w2df_dict
inverted.write_index('.', 'index')

# Upload to Bucket
os.system(f"gsutil cp index.pkl gs://{data_bucket_name}/")
os.system(f"gsutil cp DL.pkl gs://{data_bucket_name}/")
os.system(f"gsutil cp id_to_title.pkl gs://{data_bucket_name}/")

os.system(f"gsutil -m cp *.bin gs://{data_bucket_name}/")

print(f"DONE! All index files uploaded to gs://{data_bucket_name}/")

