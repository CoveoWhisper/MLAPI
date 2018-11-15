from collections import defaultdict
from pathlib import Path
from flask import json

from definitions import Definitions
import nltk

nltk.download('averaged_perceptron_tagger')

DOCUMENTS_POPULARITY_MAPPING_PATH = Path(Definitions.ROOT_DIR + "/documents_popularity.json")
DOCUMENTS_SEARCHES_MAPPING_PATH = Path(Definitions.ROOT_DIR + "/documents_searches_mapping.json")

RELATIVE_SCORES_PERFORMANCE_THRESHOLD = 0.5
MAX_PART_OF_SPEECH_SCORE = 3
PARTS_OF_SPEECH_SCORES = {
    "FW": MAX_PART_OF_SPEECH_SCORE,
    "NNP": MAX_PART_OF_SPEECH_SCORE,
    "NNPS": MAX_PART_OF_SPEECH_SCORE,
    "JJ": 1,
    "JJR": 1,
    "JJS": 1,
    "NN": MAX_PART_OF_SPEECH_SCORE,
    "NNS": MAX_PART_OF_SPEECH_SCORE,
    "RB": 1,
    "RBR": 1,
    "RBS": 1,
    "VB": 1,
    "VBD": 1,
    "VBG": 1,
    "VBN": 1,
    "VBP": 1,
    "VBZ": 1
}


def get_words_and_parts_of_speech(speech):
    return {word: part_of_speech for (word, part_of_speech) in nltk.pos_tag(speech)}


def get_searches_scores(searches_containing_context_entities, parts_of_speech_scores):
    searches_scores = defaultdict(int)
    for search, contained_context_entities in searches_containing_context_entities.items():
        words_and_parts_of_speech = get_words_and_parts_of_speech(search.split())
        score = 0
        for word, part_of_speech in words_and_parts_of_speech.items():
            if word in contained_context_entities:
                score += parts_of_speech_scores[part_of_speech] if part_of_speech in parts_of_speech_scores else 0
        searches_scores[search] = score
    return searches_scores


def get_max_score(context_entities, parts_of_speech_scores):
    context_entities_string = " ".join(context_entities)
    return get_searches_scores(
        {context_entities_string: context_entities},
        parts_of_speech_scores
    )[context_entities_string]


def get_searches_relative_scores(searches_scores, max_score):
    return {search: searches_scores[search] / max_score for search in searches_scores.keys()}


def get_searches_containing_context_entities(
        searches_documents_mapping,
        context_entities,
        max_score,
        relative_scores_performance_threshold,
        max_part_of_speech_score):
    searches_containing_context_entities = {}
    for search in searches_documents_mapping:
        context_entities_in_search = set()
        for context_entity in context_entities:
            if context_entity in search.split():
                context_entities_in_search.add(context_entity)
        if len(context_entities_in_search) * max_part_of_speech_score >= max_score * relative_scores_performance_threshold:
            searches_containing_context_entities[search] = context_entities_in_search
    return searches_containing_context_entities


def get_suggested_documents(searches_relative_scores, documents_popularity_mapping, searches_documents_mapping):
    max_relative_score = max(searches_relative_scores.values())
    max_relative_score = max(max_relative_score, 1)
    documents_relatives_scores = defaultdict(float)
    for search, documents in searches_documents_mapping.items():
        if search not in searches_relative_scores:
            continue
        for document in documents:
            if document in documents_relatives_scores and documents_relatives_scores[document] < searches_relative_scores[search]:
                documents_relatives_scores[document] = searches_relative_scores[search]
            if document not in documents_relatives_scores:
                documents_relatives_scores[document] = searches_relative_scores[search]
    documents_scores = {
        document: relative_score * documents_popularity_mapping[document]
        for document, relative_score in documents_relatives_scores.items()
        if document in documents_popularity_mapping
    }
    max_score = max(documents_scores.values())
    return {document: (score / max_score) * max_relative_score for document, score in documents_scores.items()}


def get_suggested_documents_from_analytics(context_entities):
    with open(DOCUMENTS_POPULARITY_MAPPING_PATH) as documents_popularity_mapping_file:
        documents_popularity_mapping = json.load(documents_popularity_mapping_file)
    with open(DOCUMENTS_SEARCHES_MAPPING_PATH) as documents_searches_mapping_file:
        searches_documents_mapping = json.load(documents_searches_mapping_file)
    max_score = get_max_score(context_entities, PARTS_OF_SPEECH_SCORES)
    searches_containing_context_entities = get_searches_containing_context_entities(
        searches_documents_mapping,
        context_entities,
        max_score,
        RELATIVE_SCORES_PERFORMANCE_THRESHOLD,
        MAX_PART_OF_SPEECH_SCORE
    )
    searches_scores = get_searches_scores(searches_containing_context_entities, PARTS_OF_SPEECH_SCORES)
    searches_relative_scores = get_searches_relative_scores(searches_scores, max_score)
    return get_suggested_documents(
        searches_relative_scores,
        documents_popularity_mapping,
        searches_documents_mapping
    )
