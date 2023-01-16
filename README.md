# Information_Retrieval_project:
#Information Retrieval Project: Wikipedia Search Engine

A flask app for searching relevent documents from wikipedia dumps.

## Design goals:

- Time efficiency
- Max out the mean average precision
- Retrieve relevent docs


## Docs:
All relevent information are written in files:

-  `search_frontend.py`
-  `search_backend.py`
- `Inverted_Index_gcp.py`

## Data:

- Postings anchor index (Terms are single words)
- Postings anchor double index (Terms are two-words combinations)
- Postings body index
- Postings stemm body index (Terms are stemmed before put in index)
- Postings title index
- Page view
- Page rank

## Run APP:

- run command: `python3 search_frontend.py`
