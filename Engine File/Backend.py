from flask import Flask, request, jsonify
from pathlib import Path
import pyspark
from heapq import heappop, heappush, heapify
import numpy as np
from inverted_index_gcp import *
import pickle as pickle
from collections import Counter
import nltk
import math
from threading import Thread
import math

nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter, OrderedDict
from nltk.stem.porter import *
import re

global doc_title_dic
global bucket_name
global index_title
global index_body
global index_anchor
global doc_title_dic
global bm25_T
global bm25_B

bucket_name = "buc_318237997"

# import doc_title_dic from the bucket
file_path = "doc_title_dic.pickle"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
doc_title_dic = pickle.loads(contents)

# import inverted index title from the bucket
file_path = "title1/title1_index.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
index_title = pickle.loads(contents)

# import inverted index body from the bucket
file_path = "body1/body1_index.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
index_body = pickle.loads(contents)

# import inverted index title use bm25 score from the bucket
file_path = "title_bm25/index_title_bm25.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
bm25_T = pickle.loads(contents)

# import inverted index body use bm25 score from the bucket
file_path = "body1_bm25/body1_bm25.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
bm25_B = pickle.loads(contents)

# import inverted index anchor from the bucket
bucket_name = "buc_318237997"
file_path = "anchor3/anchor3_index.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
index_anchor = pickle.loads(contents)


def get_html_pattern():
    pattern = "<(\"[^\"]\"|'[^']|[^'\">])*>"
    return pattern


def get_time_pattern():
    pattern = "((?:[01][0-2]|2[0-4])(?:\.)?(?:[0-4][0-9])|(?:[0-1]?[0-9]|2[0-5]):(?:([0-5][0-9])):(?:([0-9][0-9]))?$)((AM|am|a\.m\.|PM|p\.m\.)?$|([AP][M]|[ap]\.[m]\.))"
    return pattern


def get_number_pattern():
    pattern = "(?<![\w\+\-,\.])[\+\-]?\d{1,3}((,\d{3})|\d)(\.\d+)?(?!\S?[\w\+\-])"
    return pattern


def get_percent_pattern():
    pattern = "(?<![\w\+\-,\.])[\+\-]?\d{1,3}((,\d{3})|\d)(\.\d+)?%(?!\S?[\w\+\-])"
    return pattern


def get_date_pattern():
    pattern = "((([12][0-9]|(30)|[1-9])\ )?(Apr(il?)?|Jun(e?)?|Sep(tember?)?|Nov(ember?)?)(\ ([12][0-9],|(30,)|[1-9],))?((\ \d\d\d\d)))|((Jan(uary?)?|Mar(ch?)?|May?|Jul(y?)?|Aug(ust?)?|Oct(ober?)?|Dec(ember?)?)(\ ([12][0-9],|3[10],|[1-9],))?((\ \d\d\d\d)))|((([1][0-9]|2[0-8]|[0-9])\ )?(Feb(ruary?)?)(\ ([1][0-9],|2[0-8],|[0-9],))?((\ \d\d\d\d)))"
    return pattern


def get_word_pattern():
    pattern = "(\w+(?:-\w+)+)|(?<!-)(\w+'?\w*)"
    return pattern


RE_TOKENIZE = re.compile(rf"""
(
    # parsing html tags
     (?P<HTMLTAG>{get_html_pattern()})                                  
    # dates
    |(?P<DATE>{get_date_pattern()})
    # time
    |(?P<TIME>{get_time_pattern()})
    # Percents
    |(?P<PERCENT>{get_percent_pattern()})
    # Numbers
    |(?P<NUMBER>{get_number_pattern()})
    # Words
    |(?P<WORD>{get_word_pattern()})
    # space
    |(?P<SPACE>[\s\t\n]+) 
    # everything else
    |(?P<OTHER>.)
)
""", re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)


def filter_text(text):
    filtered = [v for match in RE_TOKENIZE.finditer(text)
                for k, v in match.groupdict().items()
                if v is not None and k not in ['HTMLTAG', 'DATE', 'TIME', 'PERCENT', 'NUMBER', 'SPACE', 'OTHER']]
    return filtered


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)


class Backend_Search:
    def __init__(self):
        self.res_body = []
        self.res_title = []
        self.candidate_body = []

    def tokenize_anchor(self, text):
        """
        This function aims in tokenize an anchor text into a list of tokens. Moreover, it filter stopwords.

        Parameters:
        -----------
        text: string , represting the text to tokenize.

        Returns:
        -----------
        list of tokens (e.g., list of tokens).
        """

        tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
        return tokens

    def tokenize(self, text):
        """
        This function aims in tokenize a text for q into a list of tokens. Moreover, it filter stopwords.
        Parameters:
        -----------
        text: string , represting the text to tokenize.
        Returns:
        -----------
        list of tokens (e.g., list of tokens).
        """

        stemmer = PorterStemmer()
        tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
        ls_lower = filter_text(' '.join(tokens))
        list_of_tokens = [token for token in ls_lower if token not in all_stopwords]
        list_res = []
        for token in list_of_tokens:
            if token not in all_stopwords:
                list_res.append(stemmer.stem(token))
        return list_res

    def read_posting_list(self, inverted, bucket_name, w, folder_name):
        """
        Read posting list of word from bucket store
        
        :param w: the requested word
        :param folder_name: name of the folder in the bucket with the bins
        :param bucket_name: name of bucket name
        :param  inverted: inverted index object
        :return posting list - list of tuple (doc_id,tf)

        """
        with closing(MultiFileReader(bucket_name, folder_name)) as reader:
            locs = inverted.posting_locs[w]
            try:
                b = reader.read(locs, inverted.df[w] * 6, bucket_name)
            except:
                return []
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * 6:i * 6 + 4], 'big')
                tf = int.from_bytes(b[i * 6 + 4:(i + 1) * 6], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def get_candidate_doc_title(self, query, index, folder_name):
        """
        :param query: list of tokens of the query
        :param folder_name: name of the folder in the bucket with the bins
        :param index: inverted index object
        :return dict:the relevant docs for the query as {doc_id:score}
        """
        candidate_docs_id = {}
        for term in np.unique(query):
            try:
                posting_list = self.read_posting_list(index, bucket_name, term, folder_name)
                for doc_id, tf in posting_list:
                    candidate_docs_id[doc_id] = candidate_docs_id.get(doc_id, 0) + 1
            except KeyError:
                continue
        return candidate_docs_id

    def get_candidate_doc_body(self, query, index, query_score):
        """
        :param query_score: the query score
        :param query: list of tokens of the query
        :param index: inverted index object
        :return dict:the docs that are relevant to the query and need to calc cosSim with them
         """
        candidate_docs_id = {}
        for term in np.unique(query):
            try:
                folder_name = "body1"
                posting_list = self.read_posting_list(index, bucket_name, term, folder_name)
                for doc_id, tf in posting_list:
                    weight = ((tf / index.weights[doc_id][0]) * math.log(index.N / index.df[term])) * query_score[term]
                    candidate_docs_id[doc_id] = candidate_docs_id.get(doc_id, 0) + weight

            except KeyError:
                pass
        return candidate_docs_id

    def search_title(self, query, pre=False):
        """
        :param pre: True if the query after tokenize else False
        :param query: the query to search
        :return list of all search results, ordered from best to
                worst where each element is a tuple (wiki_id, title).
        """
        if len(query) == 0:
            return None
        folder_name = "title1"
        if not pre:
            query = self.tokenize(query)
        scores_dict = self.get_candidate_doc_title(query, index_title, folder_name)
        if scores_dict is None:
            return []
        res,heap = [],[]
        heapify(heap)
        for key, val in scores_dict.items():
            heappush(heap, (-1 * val, key))
        while heap:
            score, doc_id = heappop(heap)
            res.append((doc_id, score * -1))
        return calc_title(res)

    def search_body(self, query, pre=False):
        """
        :param pre: True if the query after tokenize else False
        :param query: the query to search
        :return list of 100 top search results, ordered from best to
                worst where each element is a tuple (wiki_id, title).
        """
        index = index_body
        if not pre:
            query = self.tokenize(query)
        query_score = cacl_tf_idf_query(query, index)
        if len(query_score) == 0:
            return []
        candidate_docs = self.get_candidate_doc_body(query, index, query_score)
        if candidate_docs is None:
            return []
        scores_dict = cosSim(candidate_docs, index, query_score)
        res = []
        heap = []
        heapify(heap)
        for key, val in scores_dict.items():
            heappush(heap, (-1 * val, key))
        counter = 0
        while heap and counter < 100:
            score, doc_id = heappop(heap)
            res.append((doc_id, score * -1))
            counter += 1
        return calc_title(res)[:100]

    def search_anchor(self, query):
        """
        :param query: the query to search
        :return list of ALL (not just top 100) search results, ordered from best to
                worst where each element is a tuple (wiki_id, title).
        """
        folder_name = "anchor3"
        res = self.get_kind(query, index_anchor, folder_name)
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return self.calc_title_id_anchor([int(x) for x, y in res])

    def get_kind(self, query, index, folder_name):

        """
        Calculates for each doc how many tokens from query the doc has and the total frequency of token of the doc
        :param folder_name: name of the folder in the bucket with the bins
        :param index: inverted index object
        :param query: True if the query after tokenize else False
        :param query: the query to search
        :return dictionary: <doc_id, (len(tokens), freq(tokens))>
        """
        res = defaultdict(list)
        query = self.tokenize_anchor(query)
        for term in query:
            posting_list = self.read_posting_list(index, bucket_name, term, folder_name)
            for doc, freq in posting_list:
                if doc == 0:
                    continue
                res[doc].append((term, freq))
        new_res = {}
        for doc, words in res.items():
            new_res[doc] = (len(words), sum([freq for word, freq in words]))
        return new_res

    def calc_title_id_anchor(self, list_id):
        res = []
        for doc_id in list_id:
            try:
                res.append((doc_id, doc_title_dic[doc_id]))
            except KeyError:
                continue
        return res

    def backend_search(self, query):
        """
        :param query: the query to search
        :return list of ALL (not just top 100) search results, ordered by score BM25 + page_rank of doc from best to
                worst where each element is a tuple (wiki_id, title).
        """
        query = self.tokenize(query)
        if len(query) <= 2:
            text_weight, title_weight = 0.0, 1
            title_scores = self.backend_search_helper(query, bm25_T, "title_bm25")
            body_scores = self.res_body
        else:
            text_weight, title_weight = 0.6, 0.4
            threads_lst = [
                Thread(target=self.backend_search_helper, args=[query, bm25_B, "body1_bm25"]),
                Thread(target=self.backend_search_helper, args=[query, bm25_T, "title_bm25"])
            ]
            for trd in threads_lst:
                trd.start()
            threads_lst[0].join()
            threads_lst[1].join()
            body_scores = self.res_body
            title_scores = self.res_title

        merged_scores = defaultdict(float)
        if text_weight != 0.0:
            for k, v in body_scores:
                merged_scores[k] += v * text_weight
        for k, v in title_scores:
            merged_scores[k] += v * title_weight
        res = []
        heap = []
        heapify(heap)
        for key, val in merged_scores.items():
            heappush(heap, (-1 * val, key))
        counter = 0
        while heap and counter < 100:
            score, doc_id = heappop(heap)
            res.append((doc_id, score * -1))
            counter += 1
        return calc_title(res)

    def backend_search_helper(self, processed_query, index, folder):
        """
        :param folder: name of the folder in the bucket with the bins
        :param index: inverted index object
        :param processed_query: the query to search
        :return the docs that are relevant to the query as [(doc_id,score),(doc_id2,score)....]
                sorted by bm25 score in descending order
        """
        query_idf = calc_idf_query(processed_query, index)
        candidate_docs = self.get_candidate_doc_bm2_body(processed_query, index, query_idf, folder)
        if candidate_docs is None:
            return []
        res,heap = [],[]
        heapify(heap)
        for key, val in candidate_docs.items():
            heappush(heap, (-1 * val, key))
        counter = 0
        while heap and counter < 100:
            score, doc_id = heappop(heap)
            res.append((doc_id, score * -1))
            counter += 1
        if folder == "body1_bm25":
            self.res_body = res
        else:
            self.res_title = res
        return res

    def get_candidate_doc_bm2_body(self, processed_query, index, query_idf, folder_name):
        """
        :param folder_name: name of the folder in the bucket with the bins
        :param query_idf: dict of {q_term:idf}
        :param processed_query: list of tokens of the query
        :param index: inverted index object
        :return dict:the relevant docs for the query as {doc_id:bm25}
         """
        candidate_docs_id = {}
        for term in np.unique(processed_query):
            try:
                posting_list = self.read_posting_list(index, bucket_name, term, folder_name)
                for docid, tf in posting_list:
                    weight = query_idf[term] * (tf * (1.5 + 1)) / (tf + index.weights[docid]) + math.log(
                        page_rank_dict[docid])
                    candidate_docs_id[docid] = candidate_docs_id.get(docid, 0) + weight
            except KeyError:
                continue
        return candidate_docs_id


# -------------------- help function ---------------------

def calc_idf_query(query, index):
    """
    :param query:  the query to search
    :param index: inverted index object
    :return dict: dictionary of idf scores. As follows:
                                                      key: term
                                                      value: bm25 idf score
    """
    idf = {}
    for token in np.unique(query):
        N = index.N
        n = index.df.get(token, 0)
        idf[token] = math.log(((N - n + 0.5) / (n + 0.5)) + 1)
    return idf


def cosSim(candidate_docs, index, query_score):
    """
    :param query_score: dict of {query_token:tfidf weight}
    :param candidate_docs:  dict of {doc_id:numerator of cosSim}
    :param index: inverted index object
    :return query_score: returns the cosSim scores of the candidate_docs
   """
    scores = {}
    for doc_id in candidate_docs:
        scores[doc_id] = candidate_docs[doc_id] / (
            math.sqrt(index.weights[doc_id][1] * sum([math.pow(score, 2) for score in query_score.values()])))
    return scores


def calc_title(list_of_scores):
    res = []
    for doc_id, score in list_of_scores:
        try:
            res.append((doc_id, doc_title_dic[doc_id]))
        except KeyError:
            continue
    return res


def cacl_tf_idf_query(processed_query, index):
    query_score = {}
    query_tf = Counter(processed_query)
    for term in np.unique(processed_query):
        try:
            query_score[term] = (query_tf[term] / len(processed_query)) * math.log(index.N / index.df[term])
        except KeyError:
            continue
    return query_score


# --------get_page_views----------
global page_views_dict
file_path = "pageviews/pageviews-202108-user.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
page_views_dict = pickle.loads(contents)


def backend_get_page_views(list_of_pages):
    res = []
    for page in list_of_pages:
        try:
            res.append(page_views_dict[page])
        except KeyError:
            res.append(0)
    return res


# --------get_page_rank----------
global page_rank_dict
file_path = "pagerank/pagerank.pkl"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(file_path)
contents = blob.download_as_bytes()
page_rank_dict = pickle.loads(contents)


def backend_get_page_rank(list_of_pages):
    res = []
    for page in list_of_pages:
        try:
            res.append(page_rank_dict[page])
        except KeyError:
            res.append(0.0)
    return res
