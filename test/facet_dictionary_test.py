import unittest
from mlapi.facet_dictionary import FacetDictionary
from mlapi.model.facet import Facet


class TestFacetDictionary(unittest.TestCase):

    def test_create_facet_dictionary(self):
        facet_dictionary = FacetDictionary()

        tuples = [('uri1', [Facet('@y', '21'), Facet('@m', '9')]), ('uri2', [Facet('@y', '19')]), ('uri3', [Facet('@m', '9')])]
        facets_by_document = dict(tuples)

        facets = facet_dictionary.create_facet_dict(facets_by_document)

        self.assertEqual(2, len(facets))
        self.assertEqual(1, list(facets.keys()).index('@y'))
        self.assertEqual(0, list(facets.get('@y')).index('19'))
