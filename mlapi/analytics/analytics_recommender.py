import os.path
from collections import defaultdict
from pathlib import Path
from flask import json
from definitions import Definitions


DOCUMENTS_POPULARITY_MAPPING_PATH = Path(Definitions.ROOT_DIR + "/documents_popularity.json")
DOCUMENTS_SEARCHES_MAPPING_PATH = Path(Definitions.ROOT_DIR + "/documents_searches_mapping.json")
SEARCH_IMPORTANCE = 0.9
POPULARITY_IMPORTANCE = 1 - SEARCH_IMPORTANCE


def get_documents_relative_popularity(documents_popularity_mapping, popularity_importance):
    if not documents_popularity_mapping or max(documents_popularity_mapping.values()) <= 0:
        return {}
    max_popularity = max(documents_popularity_mapping.values())
    return {
        document: (popularity / max_popularity) * popularity_importance
        for document, popularity in documents_popularity_mapping.items()
    }


def get_searches_relatives_scores(searches_documents_mapping, context_entities, search_importance):
    searches_relative_scores = defaultdict(float)
    for search in searches_documents_mapping:
        search_words = search.split()
        max_score = max(2, len(search_words))
        context_entity_in_search_words = 0
        for context_entity in context_entities:
            if context_entity in search_words:
                context_entity_in_search_words += 1
        if context_entity_in_search_words > 0:
            searches_relative_scores[search] = (context_entity_in_search_words / max_score) * search_importance
    return searches_relative_scores


def get_documents_relatives_scores(searches_relative_scores, searches_documents_mapping):
    documents_relatives_scores = defaultdict(dict)
    for search, relative_score in searches_relative_scores.items():
        for document, documentTitle in searches_documents_mapping[search].items():
            if document not in documents_relatives_scores or documents_relatives_scores[document]["score"] < searches_relative_scores[search]:
                documents_relatives_scores[document]["score"] = searches_relative_scores[search]
                documents_relatives_scores[document]["title"] = documentTitle
    return documents_relatives_scores


def get_suggested_documents(documents_relatives_scores, documents_relative_popularity):
    return [
        {
            "Document": {
                "Uri": document,
                "Title": relative_score_and_title["title"]
            },
            "Score": min(relative_score_and_title["score"] + documents_relative_popularity[document], 1)
        }
        for document, relative_score_and_title in documents_relatives_scores.items()
        if document in documents_relative_popularity
    ]


class AnalyticsRecommender(object):
    def __init__(self):
        if os.path.isfile(DOCUMENTS_SEARCHES_MAPPING_PATH):
            with open(DOCUMENTS_SEARCHES_MAPPING_PATH) as documents_searches_mapping_file:
                self.searches_documents_mapping = json.load(documents_searches_mapping_file)
                documents_searches_mapping_file.close()
        else:
            self.searches_documents_mapping = {}
        if os.path.isfile(DOCUMENTS_POPULARITY_MAPPING_PATH):
            with open(DOCUMENTS_POPULARITY_MAPPING_PATH) as documents_popularity_mapping_file:
                documents_popularity_mapping = json.load(documents_popularity_mapping_file)
                documents_popularity_mapping_file.close()
        else:
            documents_popularity_mapping = {}
        self.documents_relative_popularity = get_documents_relative_popularity(documents_popularity_mapping, POPULARITY_IMPORTANCE)


    def get_suggested_documents(self, context_entities):
        if not context_entities or not self.searches_documents_mapping or not self.documents_relative_popularity:
            return []
        searches_relative_scores = get_searches_relatives_scores(self.searches_documents_mapping, context_entities, SEARCH_IMPORTANCE)
        if not searches_relative_scores:
            return []
        documents_relatives_scores = get_documents_relatives_scores(searches_relative_scores, self.searches_documents_mapping)
        if not documents_relatives_scores:
            return []
        return get_suggested_documents(documents_relatives_scores, self.documents_relative_popularity)
