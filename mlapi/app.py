import argparse

from flask import Flask, request, jsonify

from pathlib import Path
from definitions import Definitions
from mlapi.document_filter import DocumentFilter
from mlapi.facet_loader import FacetLoader
from mlapi.logger.logger_factory import LoggerFactory
from mlapi.serialization.object_encoder import ObjectEncoder
from mlapi.question_generator import QuestionGenerator
from mlapi.facet_sense_analyzer import FacetSenseAnalyzer
from mlapi.facet_sense_api import FacetSenseApi
from mlapi.facet_dictionary import FacetDictionary
from mlapi.model.facet_values import FacetValues
from mlapi.analytics.analytics_recommender import AnalyticsRecommender

FACETS_PATH = Path(Definitions.ROOT_DIR + "/facets.bin")
CREDENTIALS_PATH = Path(Definitions.ROOT_DIR + "/appsettings.json")

app = Flask(__name__)
app.json_encoder = ObjectEncoder


def initialize(facets_file, credentials_path):
    loader = FacetLoader()
    facet_dict = FacetDictionary()
    global facets_by_document
    facets_by_document = loader.load_facets(facets_file)
    global facets_dict_by_document
    facets_dict_by_document = dict()
    for document in facets_by_document:
        facet_dict = dict()
        for facet in facets_by_document[document]:
            if facet.name in facet_dict:
                facet_dict[facet.name].append(facet.value)
            else:
                facet_dict[facet.name] = [facet.value]
        facets_dict_by_document[document] = facet_dict
    global facets
    facets = facet_dict.create_facet_dict(facets_by_document)

    facet_sense_api = FacetSenseApi(credentials_path)
    global facet_sense_analyzer
    facet_sense_analyzer = FacetSenseAnalyzer(facet_sense_api)

    global analytics_recommender
    analytics_recommender = AnalyticsRecommender()


@app.route('/ML/FacetSense', methods=['POST'])
def facet_sense():
    content = request.get_json()
    analysis = facet_sense_analyzer.analyze(content['Query'])
    return jsonify(analysis)


@app.route('/ML/Analyze', methods=['POST'])
def ml_analyze():
    requested_documents = request.get_json()
    documents = dict((k, facets_by_document[k]) for k in requested_documents if k in facets_by_document)
    question_generator = QuestionGenerator()
    questions = question_generator.generate_questions(documents)
    return jsonify(questions)


@app.route('/ML/Analytics', methods=['POST'])
def analytics_analysis():
    content = request.get_json()
    suggested_documents = analytics_recommender.get_suggested_documents(content)
    return jsonify(suggested_documents)


@app.route('/ML/Filter/Facets', methods=['POST'])
def filter_document_by_facets():
    content = request.get_json()
    documents_to_filter = content['Documents']
    documents = dict((k, facets_dict_by_document[k]) for k in documents_to_filter if k in facets_dict_by_document)

    if content['MustHaveFacets'] is not None:
        must_have_facets = {val['Name']: val['Values'] for val in content['MustHaveFacets']}
        documents = DocumentFilter.keep_documents_with_facets(documents, must_have_facets)

    if content['MustNotHaveFacets'] is not None:
        must_not_have_facets = {val['Name']: val['Values'] for val in content['MustNotHaveFacets']}
        documents = DocumentFilter.keep_documents_without_facets(documents, must_not_have_facets)

    return jsonify(list(documents.keys()))


@app.route('/ML/Facets', methods=['POST'])
def get_facet_values():
    facets_name = request.get_json()
    facet_values = list()

    for name in facets_name:
        facet_values.append(FacetValues(name, facets.get(name)))

    return jsonify(facet_values)


if __name__ == '__main__':
    initialize(FACETS_PATH, CREDENTIALS_PATH)
    LoggerFactory.get_logger(__name__).info("API started")
    app.run(host='0.0.0.0')
