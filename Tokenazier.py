import re
import nltk
from nltk.corpus import stopwords #A corpus containing lists of stop words for several languages
from nltk.stem import PorterStemmer

# Initializing resources outside the functions to optimize performance during large-scale indexing.
nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenize(text):
    """
        Performs tokenization on a given text string using a predefined regular expression.

        The process includes:
        1. Converting the entire text to lowercase to ensure case-insensitivity.
        2. Finding all sequences that match the word pattern (including letters, digits,
           underscores, and optional internal hyphens or apostrophes).
        3. Limiting tokens to a maximum length of 25 characters to filter out noise.

        Args:
            text (str): The input text to be tokenized (e.g., a Wikipedia article body).

        Returns:
            list: A list of strings (tokens) extracted from the text.
    """
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
    return [token.group() for token in RE_WORD.finditer(text.lower())]

def remove_stopWord(tokens):
    """
        Filters out common English stopwords from a list of tokens.

        Args:
            tokens (list): A list of tokens (strings) to be filtered.

        Returns:
            list: A list of tokens excluding those found in the English stopwords corpus.
    """
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]
    return [t for t in tokens if t not in english_stopwords and t not in corpus_stopwords]

def stemming(tokens):
    """
        Reduces tokens to their root form (stems) using the Porter Stemmer algorithm.

        Args:
            tokens (list): A list of tokens (strings) to be stemmed.

        Returns:
            list: A list of stemmed tokens.
    """
    res = []
    for t in tokens:
        res.append(stemmer.stem(t))
    return res


def proccesc_q(querie):
    """
    Preprocesses a raw search query by applying tokenization, stopword removal, and stemming.

    This pipeline transforms a natural language string into a list of normalized tokens,
    preparing it for search against the inverted index.

    Args:
        querie (str): The raw input string from the user.

    Returns:
        list: A list of processed tokens (strings) representing the core terms of the query.

    Example:
        >>> proccesc_q("The quick brown foxes are jumping")
        ['quick', 'brown', 'fox', 'jump']
    """
    return stemming(remove_stopWord(tokenize(querie)))