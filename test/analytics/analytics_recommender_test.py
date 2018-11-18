import unittest

from mlapi.analytics.analytics_recommender import get_searches_relatives_scores, get_documents_relatives_scores, get_suggested_documents, get_documents_relative_popularity


class TestIdentifyWordsInSearches(unittest.TestCase):
    def test_get_documents_relative_popularity(self):
        documents_popularity_mapping = {
            "url1": 10,
            "url2": 20,
            "url3": 500
        }
        popularity_importance = 0.1
        expected_result = {
            "url1": (10/500) * 0.1,
            "url2": (20/500) * 0.1,
            "url3": 0.1
        }
        self.assertEqual(
            get_documents_relative_popularity(documents_popularity_mapping, popularity_importance),
            expected_result
        )

    def test_get_searches_relatives_scores(self):
        searches_documents_mapping = {
            "coveo search api dumb": "dumb",
            "coveo search api": "dumb",
            "coveo search": "dumb",
            "coveo": "dumb",
            "dumb": "dumb"
        }
        context_entities = [
            "coveo",
            "search",
            "api"
        ]
        search_importance = 0.9
        expected_result = {
            "coveo search api dumb": 0.9,
            "coveo search api": 0.9,
            "coveo search": (2/3)*0.9,
            "coveo": (1/3)*0.9
        }
        self.assertEqual(
            get_searches_relatives_scores(searches_documents_mapping, context_entities, search_importance),
            expected_result
        )

    def test_get_documents_relatives_scores(self):
        searches_relative_scores = {
            "search1": 0.9,
            "search2": 0.6
        }
        searches_documents_mapping = {
            "search1": {
                "url1": "title1"
            },
            "search2": {
                "url1": "title1",
                "url2": "title2"
            }
        }
        expected_result = {
            "url1": {
                "score": 0.9,
                "title": "title1"
            },
            "url2": {
                "score": 0.6,
                "title": "title2"
            }
        }
        self.assertEqual(
            get_documents_relatives_scores(searches_relative_scores, searches_documents_mapping),
            expected_result
        )

    def test_get_suggested_documents(self):
        documents_relatives_scores = {
            "url1": {
                "score": 0.9,
                "title": "title1"
            },
            "url2": {
                "score": 0.6,
                "title": "title2"
            },
            "url3": {
                "score": 0.9,
                "title": "title3"
            },
            "url4": {
                "score": 0.5,
                "title": "title4"
            }
        }
        documents_relative_popularity = {
            "url1": 0.1,
            "url3": 0.2,
            "url4": 0.1
        }
        expected_result = {
            "url1": {
                "score": 1,
                "title": "title1"
            },
            "url3": {
                "score": 1,
                "title": "title3"
            },
            "url4": {
                "score": 0.6,
                "title": "title4"
            }
        }
        self.assertEqual(
            get_suggested_documents(documents_relatives_scores, documents_relative_popularity),
            expected_result
        )
