import re
import os
from MediaWiki import page_iter, to_wikicode, get_wikilinks, remove_markdown, filter_article_links, get_number_pattern
from Tokenazier import tokenize, remove_stopWord, stemming


def process_wikipedia_data(file_path, limit=5):
    """
    Processes a Wikipedia dump file using an optimized pipeline.
    Filters out non-article pages and removes numeric noise using custom patterns.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Compile the complex number pattern from your MediaWiki utility
    number_pattern = re.compile(get_number_pattern(), re.UNICODE)

    print(f"--- Starting optimized processing of file: {file_path} ---")

    valid_articles_count = 0
    for doc_id, title, raw_body in page_iter(file_path):
        # Optimization: Skip non-article namespaces (Files, Talk, Wikipedia, etc.)
        if not filter_article_links(title):
            continue

        valid_articles_count += 1

        # 1. Remove MediaWiki markdown/syntax
        clean_text = remove_markdown(raw_body)

        # 2. Use your custom regex to remove numbers (e.g., "1", "4", "1,000")
        # We replace matches with a space to avoid merging adjacent words
        clean_text_no_numbers = number_pattern.sub(' ', clean_text)

        #אפשר להכניס כאן הסרת תאריכים ועוד

        # 3. Tokenize the text (now cleaned of numbers)
        tokens = tokenize(clean_text_no_numbers)

        # 4. Remove English stopwords and apply Porter Stemming
        filtered_tokens = remove_stopWord(tokens)
        stemmed_tokens = stemming(filtered_tokens)

        # Detailed output for the first 'limit' articles
        if valid_articles_count <= limit:
            print(f"\n--- Valid Article #{valid_articles_count}: {title} ---")
            print(f"Tokens count: {len(stemmed_tokens)}")
            print(f"First 10 tokens: {stemmed_tokens[:10]}")
            print("-" * 40)

        # Stop processing once the limit is reached
        if valid_articles_count >= limit:
            break

    print(f"--- Finished! Processed {valid_articles_count} articles. ---")


if __name__ == "__main__":
    # Replace with your actual file path
    process_wikipedia_data("enwiki-pages-articles.zip")