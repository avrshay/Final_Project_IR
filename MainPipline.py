import pyspark
from pyspark.sql import SparkSession
from Index import *
from inverted_index_gcp import InvertedIndex


def build_and_save_index(data_rdd, sc, output_dir, index_name):
    """
    מבצעת את כל תהליך האינדוקס: ספירת מילים, חלוקה לבאקטים ושמירה.
    """
    print(f"Starting process for: {index_name}...")

    # 1. Word Count & Grouping
    word_counts = data_rdd.flatMap(lambda x: word_count(x[0], x[1]))
    postings = word_counts.groupByKey().mapValues(reduce_word_counts)

    # 2. Filtering (רק מילים שמופיעות ביותר מ-10 מסמכים)
    postings_filtered = postings.filter(lambda x: len(x[1]) > 10)

    # 3. Calculate DF
    # הערה: הנחה ש-calculate_df מוגדרת אצלך ב-Index.py או מיובאת
    w2df = calculate_df(postings_filtered)
    w2df_dict = w2df.collectAsMap()

    # 4. Partitioning and Writing to Buckets
    locs_list = partition_postings_and_write(postings_filtered).collect()

    # 5. Merge locations
    posting_locs = merge_posting_locs(locs_list)

    # 6. Create and Save Inverted Index
    idx = InvertedIndex()
    idx.posting_locs = posting_locs
    idx.df = w2df_dict
    idx.write_index(output_dir, index_name)

    print(f"Finished {index_name}. Index saved to {output_dir}")


if __name__ == "__main__":
    # איתחול Spark
    conf = pyspark.SparkConf().setAppName("IndexingJob")
    sc = pyspark.SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()

    # טעינת הנתונים (נתיב לקובץ ה-Parquet שלך)
    path = "path/to/your/data.parquet"
    parquetFile = spark.read.parquet(path)

    # הרצה עבור Title
    title_rdd = parquetFile.limit(1000).select("title", "id").rdd
    build_and_save_index(title_rdd, sc, './title_index', 'index')

    # הרצה עבור Body
    body_rdd = parquetFile.limit(1000).select("text", "id").rdd
    build_and_save_index(body_rdd, sc, './body_indices', 'index')

    # הרצה עבור Anchor
    # הערה: וודא שיש עמודה כזו ב-Parquet או שנה בהתאם
    anchor_rdd = parquetFile.limit(1000).select("anchor_text", "id").rdd
    build_and_save_index(anchor_rdd, sc, './anchor_index', 'index')

    print("All indices have been created successfully!")