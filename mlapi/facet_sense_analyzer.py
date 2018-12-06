from mlapi.model.facet_sense_dto.facet_dto import FacetDto
from mlapi.model.facet_sense_dto.facet_value_dto import FacetValueDto
import json


class FacetSenseAnalyzer(object):
    def __init__(self, facet_sense_api):
        self.facet_sense_api = facet_sense_api

    def analyze(self, text):
        response = self.facet_sense_api.get_facet_scores(text)
        return self.content_to_facet_dtos(json.loads(response))

    def content_to_facet_dtos(self, content):
        facets = []

        if 'facets' not in content:
            return facets

        for facet in content['facets']:
            values = []
            for facet_value in facet['facetValues']:
                value = facet_value['value']
                score = facet_value['score']
                values.append(FacetValueDto(value, score))
            facets.append(FacetDto(facet['id'], facet['score'], values))
        return facets
