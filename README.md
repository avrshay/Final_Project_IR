# Wikipedia Search Engine

## Backend Documentation:

## Overview
This backend implements the core of a large-scale search engine for Wikipedia articles using Spark on Google Cloud Platform (GCP). The system processes over 6 million Wikipedia pages to create an inverted index with BM25 ranking, PageRank scoring, and additional search capabilities.

## File Structure

### 1. Backend.ipynb
Main implementation notebook containing the complete data processing pipeline.

### 2. inverted_index_gcp.py
Helper module providing the `InvertedIndex` class and utilities for reading/writing posting lists to Google Cloud Storage.

## Backend.ipynb - Key Components

### Setup Section
- **Cluster Configuration**: Initializes a Google Dataproc cluster with 2 primary workers
- **Dependencies**: Installs required libraries (google-cloud-storage, graphframes)
- **Imports**: Loads Spark, NLTK, and custom utilities
- **Data Loading**: Connects to GCS bucket and loads Wikipedia parquet files

### Inverted Index Creation

**Text Preprocessing**:
- `word_count()`: Tokenizes text, removes stopwords, counts term frequencies
- Uses regex pattern `RE_WORD` for token extraction
- Filters out English stopwords + Wiki-corpus-specific stopwords

**Index Pipeline**:
1. Maps documents to (term, (doc_id, tf)) pairs
2. Groups by term and reduces to sorted posting lists
3. Filters terms appearing in fewer than 50 documents
4. Calculates document frequency (df) for each term
5. Partitions posting lists into 124 buckets using hash-based distribution
6. Writes posting lists to disk in binary format

**Output**: 
- `index.pkl`: Contains df dictionary and posting list locations
- Binary posting list files in GCS

### Document Length (DL) Calculation
Computes total token count per document for BM25 scoring:
- Aggregates term frequencies per document
- Saves as compressed CSV to the bucket in 'dl'

### ID-to-Title Mapping
Extracts and stores document ID to title mappings:
- Selects non-null titles from parquet files
- Saves as compressed CSV to the bucket in 'id_to_title'

### PageRank Computation

**Graph Construction**:
- `generate_graph()`: Builds directed graph from Wikipedia links
- Creates edges from anchor text references
- Removes duplicate edges between same page pairs

**PageRank Execution**:
- Uses GraphFrames library on Spark
- Parameters: reset probability = 0.15, max iterations = 6
- Outputs sorted by PageRank score
- Saves as compressed CSV to the bucket in 'pr'

## inverted_index_gcp.py - Core Classes

### InvertedIndex
Main class for managing the inverted index structure.

**Key Attributes**:
- `df`: Document frequency per term
- `term_total`: Total frequency per term across corpus
- `posting_locs`: File locations for each term's posting list

**Key Methods**:
- `write_index()`: Serializes index metadata to pickle file
- `read_a_posting_list()`: Retrieves posting list for a specific term from GCS
- `posting_lists_iter()`: Generator for iterating through all posting lists

### MultiFileWriter
Handles writing large posting lists across multiple fixed-size binary files (BLOCK_SIZE = 1999998 bytes).

**Functionality**:
- Automatically creates new files when current file reaches size limit
- Returns list of (filename, offset) tuples for each write operation
- Supports both local filesystem and GCS buckets

### MultiFileReader
Reads posting lists from multiple binary files.

**Functionality**:
- Maintains open file handles for efficient random access
- Reads from specific (filename, offset) locations
- Handles cross-file reads when posting lists span multiple files


## Integration Points

The backend interfaces with the search frontend through:
1. **InvertedIndex class**: Loads index and retrieves posting lists
2. **DL file**: Provides document lengths for BM25 calculation
3. **ID-Title mapping**: Converts document IDs to readable titles
4. **PageRank scores**: Enhances ranking with link-based authority

### - Performance Optimizations at the backend -

- Hash-based partitioning distributes terms evenly across 124 buckets
- Filtering removes rare terms (< 50 documents) to reduce index size
- Spark RDD operations enable parallel processing across cluster


## Frontend Documentation:

## Overview
This frontend provides a RESTful API for the Wikipedia Search Engine. It acts as the interface between users and the pre-processed indices, implementing various search retrieval methods (BM25, PageRank integration) and utility endpoints for document metadata.

## File Structure

### 1. search_frontend.py
The main Flask application containing the API endpoints and retrieval logic.

### 2. inverted_index_gcp.py
We have explained above


## search_frontend.py - Key Components

### Setup Section
- **Flask App Setup**: Initializes a `MyFlaskApp` instance running on port 8080
- **Resource Loading**: Loads critical data structures from the local filesystem (`/home/moriyc`):
    - `index.pkl`: The inverted index metadata
    - `DL_dict.pkl`: Document lengths for BM25 normalization
    - `pr_dict.pkl`: PageRank scores for authority-based ranking
    - `ID2TITLE_dict.pkl`: Mapping of Wikipedia IDs to their corresponding titles
- **Stopwords**: Combines standard NLTK English stopwords with Wikipedia-specific noise words (e.g., "category", "thumb", "links")

  
### BM25_from_index - Core Class
Main class for implementing the Best Match 25 ranking algorithm.

**Key Attributes**:
- `k1`, `b`: Free parameters for term frequency saturation and length normalization (default $k1=1.5, b=0.75$)
- `DL`: Dictionary of document lengths for normalization
- `index`: Reference to the `InvertedIndex` object for accessing metadata

**Key Methods**:
- `calc_idf()`: Calculates the Inverse Document Frequency for query terms using the formula: $ln(1 + \frac{N - n(t_i) + 0.5}{n(t_i) + 0.5})$
- `search()`: Implements Term-at-a-Time (TAAT) scoring, retrieving specific posting lists and accumulating scores in a `Counter` object
- `search_with_pagerank()`: Combines BM25 content scores with PageRank authority scores using a weighted linear combination ($\alpha$)

### Search Implementation

**Hybrid Ranking**:
- `search_with_pagerank()`: Combines BM25 content scores with PageRank authority scores using a weighted linear combination ($\alpha$)
- Processes queries through the same `RE_WORD` regex pattern to ensure consistency with the backend processing

**API Endpoints**:
- `POST /get_pagerank`: Accepts a list of article IDs and returns their corresponding PageRank scores


## Integration Points

The frontend interfaces with the pre-processed backend data through:
1. **InvertedIndex class**: Loads index metadata and uses `read_a_posting_list` to fetch binary data from `.bin` files
2. **Local File Access**: Reads dictionaries and indices directly from the local disk for low-latency API responses
3. **Consistent Tokenization**: Uses the identical preprocessing pipeline (regex and stopwords) as the backend to ensure query terms match indexed terms

### - Performance Optimizations at the frontend -

- Lazy Loading: Posting lists are only read from disk when a specific term is queried to minimize memory footprint
- Memory-Mapped Lookups: Pre-computed dictionaries (DL, PageRank, Titles) are loaded into memory for $O(1)$ access speed
- Efficient Aggregation: Utilizes Python `Counter` objects for high-speed score accumulation during the ranking phase
- Local I/O: Accessing binary files from local storage instead of GCS to minimize network overhead and latency
