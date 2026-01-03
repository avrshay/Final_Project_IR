# Parsing articles (MediaWiki) - from assignment 1:
import bz2
import re
from xml.etree.ElementTree import ElementTree
import mwparserfromhell as mwp  # Python library for parsing Wikipedia's markup language (Wikitext).

def page_iter(wiki_file):
    """
        Reads a wiki dump file and creates a generator that yields pages.
        This function efficiently processes massive XML files by yielding one page at a time
        instead of loading the entire file into memory.

        Args:
            wiki_file (str): A path to a compressed (.bz2) wiki dump file.

        Yields:
            tuple: (article_id, title, body) for each article.
    """
    # open compressed bz2 dump file
    with bz2.open(wiki_file, 'rt', encoding='utf-8', errors='ignore') as f_in:
        # Instead of loading the entire massive XML file into memory at once, this iterator
        # yields one piece at a time (working from the inside out) as soon as it finishes reading a tag.

        elems = (elem for _, elem in ElementTree.iterparse(f_in, events=("end",)))

        # Consume the first element and extract the xml namespace from it.
        elem = next(elems)
        t = elem.tag
        ns = t[t.find('{'):t.find('}') + 1]

        for elem in elems:
            if elem.tag == f'{ns}page':
                title = elem.find(f'{ns}title').text
                # Redirect pages should be ignored for indexing purposes
                if elem.find(f'{ns}redirect') is not None:
                    continue

                # Extracting ID and Body
                id = elem.find(f'{ns}id').text
                body = elem.find(f'{ns}revision/{ns}text').text
                yield id, title, body

                # Clear the element from memory once it has been processed to prevent memory leaks.
                elem.clear()
def remove_markdown(text):
    """
        Parses the raw MediaWiki text and removes all technical markup (like templates {{...}} and internal links [[...]]),
        returning only the clean, readable plain text.

        Args:
            text (str): The raw MediaWiki text from a Wikipedia article.

        Returns:
            str: A clean string containing only the text content of the article.
        """
    return mwp.parse(text).strip_code()
def to_wikicode(rawText):
    """
        Converts raw MediaWiki text into a 'Wikicode' object (a parse tree).

        Args:
            rawText (str): The raw string content of a Wikipedia article.

        Returns:
            mwparserfromhell.wikicode.Wikicode: A parse tree object representing the article.
    """
    return mwp.parse(rawText)
def filter_article_links(title):
    """
    Filters out links that do not point to main Wikipedia articles.

    Args:
        title (str): The destination title of the link.

    Returns:
        bool: True if the link is a valid article, False otherwise.
    """
    # List of namespaces that should not be indexed
    blacklist = [
        'media',
        'special',

        'talk',
        'user',
        'user talk',
        'wikipedia',
        'wikipedia talk',
        'file',
        'file talk',
        'mediawiki',
        'mediawiki talk',
        'template',
        'template talk',
        'help',
        'help talk',
        'category',
        'category talk',

        'portal',
        'portal talk',
        'draft',
        'draft talk',
        'timedtext',
        'timedtext talk',
        'module',
        'module talk',

        'book',
        'book talk',
        'gadget',
        'gadget talk',
        'gadget definition',
        'gadget definition talk',
        'topic',

        'w',
        'm',
        'wp',
        'wt',
        'image',
        'image talk',
        'project',
        'project talk',

        'wikt', 'wiktionary',
        'commons',
        'meta',
        'species',
        'wikibooks', 'b',
        'wikinews', 'n',
        'wikiquote', 'q',
        'wikisource', 's',
        'wikiversity', 'v',
        'wikivoyage', 'voy',
        'wikidata', 'd',

        'fr', 'de', 'es', 'it', 'ru', 'ja', 'zh', 'pt', 'pl', 'nl', 'ar', 'he',
    ]

    # If there is no colon, it is a main article (no prefix)
    if ":" not in title:
        return True

    prefix = title.split(':')[0]
    # Remove whitespace and convert to lowercase for comparison
    namespace = prefix.strip().lower()

    if namespace in blacklist:
        return False

    return True
def build_id_title_map(wiki_file):
    """
    Iterates through the Wikipedia dump and creates a dictionary mapping
    each Article ID to its Title.

    The map is saved as a pickle file for fast loading in the search frontend.

    Args:
        wiki_file (str): Path to the bz2 wikipedia dump.

    Returns:
        dict: A dictionary where the keys are Article IDs (int) and the values
        are the corresponding article Titles (str).
    """
    id_title_map = {}

    for doc_id, title, _ in page_iter(wiki_file):
        # Convert doc_id to int to save memory and ensure consistency
        id_title_map[int(doc_id)] = title

    return id_title_map
def get_wikilinks(wikicode):
    """
        Traverses the parse tree for internal links and filters out non-article links.

        Args:
            wikicode: mwparserfromhell.wikicode.Wikicode
                The parsed tree object of a Wikipedia article's markdown.

        Returns:
            list: A list of (link_title: str, anchor_text: str) pairs representing
                  outgoing links to other Wikipedia articles.
    """
    links = []
    for wl in wikicode.ifilter_wikilinks():
        # skip links that don't pass our filter (e.g., non-article namespaces)
        title = str(wl.title)
        if not filter_article_links(title):
            continue

        # Retrieve the anchor text
        text = wl.text
        if text is None:
            # Case where only the title exists without specific anchor text (e.g., [[Apple]])
            text = title
        else:
            # Convert anchor text to a clean string without special markup (e.g., [[Apple|The Fruit]])
            text = text.strip_code()

        # Remove any lingering section/anchor reference in the link (e.g., 'Apple#History' -> 'Apple')
        title = title.split('#')[
            0]
        text = text.split('#')[
            0]

        links.append((title, text))
    return links

#Get functions for regex patterns:
def get_html_pattern():
    """
    Generates a regular expression pattern to identify and match HTML tags.

    Returns:
        str: A regex string for matching HTML tags like <div>, <p>, etc.
    """
    return r"<.*?>"
def get_date_pattern():
    """
        Generates a comprehensive regex pattern for various English date formats.

        Covers four main styles:
        1. Textual: 'Month Day, Year' (e.g., January 31, 2025).
        2. Textual: 'Day Month Year' (e.g., 31 January 2025).
        3. Numeric: 'DD/MM/YYYY' or 'DD-MM-YYYY'.
        4. Numeric: 'MM/DD/YYYY' or 'MM-DD-YYYY'.

        It includes logic for month lengths (e.g., Feb up to 28 days) and
        common abbreviations (Jan, Feb, etc.).

        Returns:
            str: A complex regex string for date identification.
    """
    long_months = r"(January|March|May|July|August|October|December|Jan|Mar|May|Jul|Aug|Oct|Dec)"

    short_months = r"(April|June|September|November|Apr|Jun|Sep|Nov)"

    feb = r"(February|Feb)"

    day_31 = r"([1-9]|[12][0-9]|3[01])"
    day_30 = r"([1-9]|[12][0-9]|30)"
    day_feb = r"([1-9]|1[0-9]|2[0-8])"

    year = r"\d{1,4}"

    pattern1 = (
        rf"(?:"
        rf"{long_months}\s{day_31},\s{year}"
        rf"|{short_months}\s{day_30},\s{year}"
        rf"|{feb}\s{day_feb},\s{year}"
        rf")"
    )

    pattern2 = (
        rf"(?:"
        rf"{day_31}\s{long_months}\s{year}"
        rf"|{day_30}\s{short_months}\s{year}"
        rf"|{day_feb}\s{feb}\s{year}"
        rf")"
    )

    numeric_day = r"(?:0?[1-9]|[12][0-9]|3[01])"
    numeric_month = r"(?:0?[1-9]|1[0-2])"
    numeric_year = r"\d{1,4}"
    pattern3 = rf"{numeric_day}[-/.]{numeric_month}[-/.]{numeric_year}"
    pattern4 = rf"{numeric_month}[-/.]{numeric_day}[-/.]{numeric_year}"

    return rf"({pattern1}|{pattern2}|{pattern3}|{pattern4})"

def get_time_pattern():
    """
        Generates a regex pattern for time strings in 12-hour or 24-hour formats.

        Matches:
        - H:M:S (e.g., 14:30:05).
        - HH.MM or HHMM followed by AM/PM suffixes (e.g., 02.30PM, 11a.m.).

        Uses lookbehind and lookahead to ensure the time is not part of a
        larger alphanumeric string.

        Returns:
            str: A regex string for time identification.
    """
    hours_h = r"(2[0-3]|[0-1]?[0-9])"

    hours_hh = r"(2[0-3]|[0-1][0-9])"

    minutes_mm = r"[0-5][0-9]"

    seconds_ss = r"[0-5][0-9]"

    suffix_part = r"(AM|PM|a\.m\.|p\.m\.)"

    # H:M:S
    opt1 = rf"({hours_h}:{minutes_mm}:{seconds_ss})"

    # HH.MM or HHMM or with suffix
    opt2 = rf"({hours_hh}\.?{minutes_mm}{suffix_part})"

    start = r"((?<=^)|(?<![A-Za-z0-9]))"

    end = r"(?![A-Za-z0-9])"

    return rf"{start}({opt1}|{opt2}){end}"

def get_percent_pattern():
    """
        Generates a regex pattern for matching percentage values.

        Handles:
        - Positive/negative signs (+5%, -10.5%).
        - Thousands separators (1,000%).
        - Decimals (99.99%).

        Ensures that the '%' sign is at the end and not followed by
        irrelevant punctuation or characters.

        Returns:
            str: A regex string for percentage identification.
    """
    sign_part = r"[+-]?"

    part_with_commas = r"\d{1,3}(,\d{3})+"

    part_no_commas = r"\d+"

    dec_part = r"(\.\d+)?"

    opt1 = rf"({part_with_commas}{dec_part})"
    opt2 = rf"({part_no_commas}{dec_part})"

    # Either start of string, or no forbidden char before
    start = r"((?<=^)|(?<![A-Za-z0-9_%+\-.,]))"

    # Must end with percent & not be followed by letter, digit, comma+digit, or dot+letter/digit
    end = r"%(?![A-Za-z0-9_]|,\d|\.[A-Za-z\d])"

    return rf"{start}{sign_part}({opt1}|{opt2}){end}"

def get_number_pattern():
    """
        Generates a regex pattern for matching numerical values.
        Matches integers, decimals, and numbers with thousands-separators (commas).
        Includes logic to ignore numbers that are part of other entities like
        dates or IDs by using boundary constraints.

        Returns:
            str: A regex string for general number identification.
    """
    sign_part = r"[+-]?"

    part_with_commas = r"\d{1,3}(,\d{3})+"

    part_no_commas = r"\d+"

    dec_part = r"(\.\d+)?"

    opt1 = rf"({part_with_commas}{dec_part})"
    opt2 = rf"({part_no_commas}{dec_part})"

    # Either start of string, or no forbidden char before
    start = r"((?<=^)|(?<![A-Za-z0-9_%+\-.,]))"

    # Must not be followed by letter, digit, percent, comma+digit, or dot+letter/digit
    end = r"(?![A-Za-z0-9_%]|,\d|\.[A-Za-z\d])"

    return rf"{start}{sign_part}({opt1}|{opt2}){end}"

def get_word_pattern():
    """
        Generates a regex pattern for standard words and hyphenated terms.

        Matches:
        - Basic words (e.g., 'Apple').
        - Hyphenated words (e.g., 'ice-cream', 'well-known').
        - Words with internal apostrophes (e.g., 'don't', 'it's').
        - Optional surrounding punctuation (brackets, quotes).

        Returns:
            str: A regex string for general word identification.
    """
    start = r"(?<![-'])"

    optional = r"[('\"[{]?"

    letters = r"[A-Za-z]+(-[A-Za-z]+)*"  # can seperate with '-'

    option_beforeLast = r"(?:'?[A-Za-z])"  # ' before last letter

    ending = r"(['\"\)\]\}])?"

    return rf"{start}{optional}{letters}{option_beforeLast}{ending}"


