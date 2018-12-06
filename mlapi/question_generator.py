from collections import Counter
from mlapi.model.question import Question
from mlapi.discriminating_algo import DiscriminatingFacetsAlgo
import uuid


class QuestionGenerator(object):
    def generate_questions(self, facets_by_document):
        if not facets_by_document:
            return []

        values_by_name = self.get_discriminating_facets(facets_by_document)
        questions = []
        for facet in values_by_name:
            questions.append(Question(id=str(uuid.uuid4()), facet_name=facet[0], facet_values=list(set(facet[1])), answer="", status="None"))
        return questions

    def get_discriminating_facets(self, facets_by_document):
        unique_facets_by_document = self.remove_redundancies_in_documents(facets_by_document)
        algorithm = DiscriminatingFacetsAlgo()
        return algorithm.get_discriminating_facets(unique_facets_by_document)

    def remove_redundancies_in_documents(self, facets_by_document):
        redundancies = self.get_redundancies(facets_by_document)
        return self.remove_redundant_values(facets_by_document, redundancies)

    def remove_redundant_values(self, facets_by_document, redundancies):
        unique_facets_by_document = {}
        for document, facets in facets_by_document.items():
            facet_set = set(facets)
            unique_facets = [x for x in facet_set if x not in redundancies]
            if len(unique_facets) is not 0:
                unique_facets_by_document[document] = unique_facets
        return unique_facets_by_document

    def get_redundancies(self, facets_by_document):
        all_facets = []
        for document, facets in facets_by_document.items():
            all_facets += facets
        facet_count_by_facet = Counter(all_facets)
        return set([facet for facet in facet_count_by_facet if facet_count_by_facet[facet] == len(facets_by_document.keys())])
