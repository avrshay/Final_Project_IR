from collections import Counter
import pickle
import os


def get_pageviwe():
    """
    מייצר מילון הממפה מזהה מאמר למספר צפיות מתוך הקובץ המעובד.
    שימו לב: יש להריץ את פקודות ה-Shell שסופקו לכם כדי לייצר את pv_temp קודם לכן.
    """
    wid2pv = Counter()

    pv_temp = 'pageviews-202108-user-4dedup.txt'  # create in shell

    if not os.path.exists(pv_temp):
        print(f"Error: {pv_temp} not found. Make sure to run the shell commands first.")
        return wid2pv

    with open(pv_temp, 'rt') as f:
        for line in f:
            parts = line.split(' ')
            if len(parts) == 2:
                wid2pv.update({int(parts[0]): int(parts[1])})

    return wid2pv


def most_viewed(pages):
    """Rank pages from most viewed to least viewed using the above `wid2pv`
     counter.
  Parameters:
  -----------
    pages: An iterable list of pages as returned from `page_iter` where each
           item is an article with (id, title, body)
  Returns:
  --------
  A list of tuples
    Sorted list of articles from most viewed to least viewed article with
    article title and page views. For example:
    [('Langnes, Troms': 16), ('Langenes': 10), ('Langenes, Finnmark': 4), ...]
  """

    temp_list = []
    pv = get_pageviwe(pages)

    for (page_id, page_title, page_body) in pages:
        views = pv.get(str(page_id), 0)  # getting the num of views, if doesn't exisit it will get 0

        temp_list.append((page_title, views))

    sol = sorted(temp_list, key=lambda item: item[1], reverse=True)

    return sol
