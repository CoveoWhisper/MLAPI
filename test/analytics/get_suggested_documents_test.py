import unittest

from mlapi.analytics.get_suggested_documents import get_words_and_parts_of_speech, \
    get_searches_containing_context_entities, get_searches_scores, get_searches_relative_scores, \
    get_suggested_documents, get_max_score


class TestIdentifyWordsInSearches(unittest.TestCase):
    def test_get_max_score(self):
        context_entities = ["throw", "banana"]
        parts_of_speech_scores = {
            "NN": 2,
        }
        self.assertEqual(get_max_score(context_entities, parts_of_speech_scores), 4)

    def test_get_max_score_count_duplicates_only_once(self):
        context_entities = ["throw", "banana", "banana"]
        parts_of_speech_scores = {
            "NN": 2,
        }
        self.assertEqual(get_max_score(context_entities, parts_of_speech_scores), 4)

    def test_get_searches_containing_context_entities_find_searches_with_at_least_one_same_word(self):
        searches_documents_mapping = {
            "i am a key": [
                "i am a link"
            ],
            "another key": [
                "another link"
            ],
            "no same word": [
                "another link"
            ]
        }
        context_entities = {"key"}
        expected_result = {
            "i am a key": {
                "key"
            },
            "another key": {
                "key"
            }
        }
        actual_result = get_searches_containing_context_entities(searches_documents_mapping, context_entities, 1, 1, 3)
        self.assertEqual(actual_result, expected_result)

    def test_get_searches_containing_context_entities_returns_all_same_words(self):
        searches_documents_mapping = {
            "i am a key": [
                "i am a link"
            ],
            "another key": [
                "another link"
            ],
            "no same word": [
                "another link"
            ]
        }
        context_entities = {"key", "another", "a"}
        expected_result = {
            "i am a key": {
                "key",
                "a"
            },
            "another key": {
                "key",
                "another"
            }
        }
        actual_result = get_searches_containing_context_entities(searches_documents_mapping, context_entities, 1, 1, 3)
        self.assertEqual(actual_result, expected_result)

    def test_get_searches_containing_context_entities_returns_only_searches_that_can_have_a_score_over_threshold(self):
        searches_documents_mapping = {
            "i am a key": [
                "i am a link"
            ],
            "another key": [
                "another link"
            ],
            "no same word but key": [
                "another link"
            ]
        }
        context_entities = {"key", "another", "a"}
        expected_result = {
            "i am a key": {
                "key",
                "a"
            },
            "another key": {
                "key",
                "another"
            }
        }
        actual_result = get_searches_containing_context_entities(searches_documents_mapping, context_entities, 6, 1, 3)
        self.assertEqual(actual_result, expected_result)

    def test_identify_words_in_searches(self):
        test_search = ["throw", "your", "banana"]
        expected_results = {
            "banana": "NN",
            "your": "PRP$",
            "throw": "VB"
        }
        self.assertEqual(get_words_and_parts_of_speech(test_search), expected_results)

    def test_get_search_score(self):
        searches_containing_context_entities = {
            "throw your banana": {
                "throw",
                "banana"
            },
            "you love banana": {
                "banana",
                "you"
            }
        }
        parts_of_speech_scores = {
            "NN": 5,
            "NNS": 5,
            "VB": 2,
            "VBP": 2,
            "PRP": 2
        }
        expected_search_scores = {
            "throw your banana": 7,
            "you love banana": 7
        }
        self.assertEqual(
            expected_search_scores,
            get_searches_scores(searches_containing_context_entities, parts_of_speech_scores)
        )

    def test_get_search_score_with_duplicates_only_counts_word_once(self):
        searches_containing_context_entities = {
            "throw your banana and banana": {
                "throw",
                "banana"
            }
        }
        parts_of_speech_scores = {
            "NN": 5,
            "VB": 2,
        }
        expected_search_scores = {
            "throw your banana and banana": 7
        }
        self.assertEqual(
            expected_search_scores,
            get_searches_scores(searches_containing_context_entities, parts_of_speech_scores)
        )

    def test_get_search_score_only_count_words_in_context_entities(self):
        searches_containing_context_entities = {
            "throw your banana": {
                "throw",
                "banana"
            }
        }
        parts_of_speech_scores = {
            "NN": 5,
            "VB": 2,
            "PRP$": 42
        }
        expected_search_scores = {
            "throw your banana": 7
        }
        self.assertEqual(
            expected_search_scores,
            get_searches_scores(searches_containing_context_entities, parts_of_speech_scores)
        )

    def test_get_searches_relative_scores(self):
        search_scores = {
            "throw your banana": 4,
            "throw your apple": 2
        }
        expected_search_scores = {
            "throw your banana": 1,
            "throw your apple": 1/2
        }
        self.assertEqual(
            expected_search_scores,
            get_searches_relative_scores(search_scores, 4)
        )

    def test_get_suggested_documents(self):
        searches_relative_scores ={
            "throw your banana": 1,
            "throw your apple": 1/2,
            "throw your orange": 1/2,
            "don't do that": 1/4
        }
        documents_popularity_mapping = {
            "a": 90,
            "b": 50,
            "c": 2,
            "d": 98,
            "e": 25,
            "f": 200,
            "g": 1000,
            "h": 800,
            "i": 800
        }
        searches_documents_mapping = {
            "throw your banana": ["a","b","c","y"],
            "throw your apple": ["d","e","f"],
            "don't do that": ["g"],
            "please no": ["h"],
            "again!?": ["z"]
        }
        expected_suggested_documents = {
            'a': 0.36,
            'b': 0.2,
            'c': 0.008,
            'd': 0.196,
            'e': 0.05,
            'f': 0.4,
            'g': 1.0
        }
        self.assertEqual(
            expected_suggested_documents,
            get_suggested_documents(
                searches_relative_scores,
                documents_popularity_mapping,
                searches_documents_mapping
            )
        )
