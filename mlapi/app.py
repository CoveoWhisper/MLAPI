from flask import Flask, request, jsonify

from pathlib import Path
from definitions import Definitions
from mlapi.document_filter import DocumentFilter
from mlapi.facet_loader import FacetLoader
from mlapi.logger.logger_factory import LoggerFactory
from mlapi.serialization.object_encoder import ObjectEncoder
from mlapi.question_generator import QuestionGenerator


FACETS_FILE = Path(Definitions.ROOT_DIR + "/facets.bin")

app = Flask(__name__)
app.json_encoder = ObjectEncoder
loader = FacetLoader()
facets_by_document_temp = loader.load_facets(FACETS_FILE)
facets_by_document = dict()
for document in facets_by_document_temp:
    facet_dict = dict()
    for facet in facets_by_document_temp[document]:
        if facet.name in facet_dict:
            facet_dict[facet.name].append(facet.value)
        else:
            facet_dict[facet.name] = [facet.value]

    facets_by_document[document] = facet_dict

@app.route('/ML/Analyze', methods=['POST'])
def ml_analyze():
    requested_documents = request.get_json()
    documents = dict((k, facets_by_document[k]) for k in requested_documents if k in facets_by_document)
    question_generator = QuestionGenerator()
    questions = question_generator.generate_questions(documents)
    return jsonify(questions)


@app.route('/ML/Filter/Facets', methods=['POST'])
def filter_document_by_facets():
    content = request.get_json()
    documents_to_filter = content['Documents']
    documents = dict((k, facets_by_document[k]) for k in documents_to_filter if k in facets_by_document)

    if content['MustHaveFacets'] is not None:
        must_have_facets = {val['Name']: val['Values'] for val in content['MustHaveFacets']}
        documents = DocumentFilter.keep_documents_with_facets(documents, must_have_facets)

    if content['MustNotHaveFacets'] is not None:
        must_not_have_facets = {val['Name']: val['Values'] for val in content['MustNotHaveFacets']}
        documents = DocumentFilter.keep_documents_without_facets(documents, must_not_have_facets)

    return jsonify(list(documents.keys()))


if __name__ == '__main__':
    LoggerFactory.get_logger(__name__).info("API started")
    app.run(host='0.0.0.0')
