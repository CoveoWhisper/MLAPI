import unittest
from mlapi.model.facet import Facet
from mlapi.question_generator import DiscriminatingFacetsAlgo
import statistics

class TestDiscriminatingFacets(unittest.TestCase):
    def test_when_no_discriminating_facet_then_return_no_facet(self):
        discriminating_algo = DiscriminatingFacetsAlgo()
        facet_a1 = Facet('NameA', 'ValueA1')
        facet_a2 = Facet('NameA', 'ValueA2')
        facet_b = Facet('NameB', 'ValueB')
        facets_by_document = {'1': [facet_a1], '2': [facet_a2], '3': [facet_b]}

        discriminating_facets = discriminating_algo.get_discriminating_facets(facets_by_document)
        self.assertEqual(len(discriminating_facets), 0)

    def test_when_1_discriminating_facet_then_rerun_algorithm_and_return_1_facet(self):
        discriminating_algo = DiscriminatingFacetsAlgo()
        facet_a1 = Facet('NameA', 'ValueA1')
        facet_a2 = Facet('NameA', 'ValueA2')
        facet_b = Facet('NameB', 'ValueB')
        facets_by_document = {'1': [facet_a1], '2': [facet_a2], '3': [facet_a2], '4': [facet_b]}

        discriminating_facets = discriminating_algo.get_discriminating_facets(facets_by_document)
        self.assertEqual(len(discriminating_facets), 1)
        self.assertEqual(discriminating_facets[0][0], 'NameA')
        self.assertEqual(discriminating_facets[0][1], ['ValueA1', 'ValueA2'])

    def test_when_2_discriminating_facets_then_run_one_time_algorithm_and_return_2_facets(self):
        discriminating_algo = DiscriminatingFacetsAlgo()
        facet_a1 = Facet('NameA', 'ValueA1')
        facet_a2 = Facet('NameA', 'ValueA2')
        facet_b = Facet('NameB', 'ValueB')
        facet_c1 = Facet('NameC', 'ValueC1')
        facet_c2 = Facet('NameC', 'ValueC2')
        facet_c3 = Facet('NameC', 'ValueC3')
        facets_by_document = {'1': [facet_a1], '2': [facet_a2, facet_c1], '3': [facet_a1, facet_c2, facet_c3], '4': [facet_b, facet_c3]}

        discriminating_facets = discriminating_algo.get_discriminating_facets(facets_by_document)
        self.assertEqual(len(discriminating_facets), 2)
        self.assertEqual(discriminating_facets[0][0], 'NameC')
        self.assertEqual(discriminating_facets[0][1], ['ValueC1', 'ValueC2', 'ValueC3'])
        self.assertEqual(discriminating_facets[1][0], 'NameA')
        self.assertEqual(discriminating_facets[1][1], ['ValueA1', 'ValueA2'])

    def test_when_facet_dont_have_3_documents_then_return_0_facet(self):
        discriminating_algo = DiscriminatingFacetsAlgo()
        facet_a1 = Facet('NameA', 'ValueA1')
        facet_a2 = Facet('NameA', 'ValueA2')
        facet_a3 = Facet('NameA', 'ValueA3')
        facet_a4 = Facet('NameA', 'ValueA4')
        facets_by_document = {'1': [facet_a1, facet_a2], '2': [facet_a3, facet_a4]}

        discriminating_facets = discriminating_algo.get_discriminating_facets(facets_by_document)
        self.assertEqual(len(discriminating_facets), 0)

    def test_when_facet_dont_have_2_values_then_return_0_facet(self):
        discriminating_algo = DiscriminatingFacetsAlgo()
        facet_a = Facet('NameA', 'ValueA')
        facet_b = Facet('NameB', 'ValueB')
        facets_by_document = {'1': [facet_a], '2': [facet_a], '3':[facet_b], '4':[facet_a]}

        discriminating_facets = discriminating_algo.get_discriminating_facets(facets_by_document)
        self.assertEqual(len(discriminating_facets), 0)

    def test_when_document_counts_in_facet_values_have_more_than_35_in_standard_deviation_then_return_0_facet(self):
        discriminating_algo = DiscriminatingFacetsAlgo()
        facet_a1 = Facet('NameA', 'ValueA1')
        facet_a2 = Facet('NameA', 'ValueA2')
        facet_b = Facet('NameB', 'ValueB')
        facets_by_document = {'1': [facet_a1],
                              '2': [facet_a2], '3':[facet_a2], '4': [facet_a2], '5': [facet_a2], '6':[facet_a2],
                              '7': [facet_a2], '8': [facet_a2], '9': [facet_a2], '10': [facet_a2], '11': [facet_a2],
                              '12': [facet_a2], '13': [facet_a2], '14': [facet_a2], '15': [facet_a2], '16': [facet_a2],
                              '17': [facet_a2], '18': [facet_a2], '19': [facet_a2], '20': [facet_a2], '21': [facet_a2],
                              '22': [facet_a2], '23': [facet_a2], '24': [facet_a2], '25': [facet_a2], '26': [facet_a2],
                              '27': [facet_a2], '28': [facet_a2], '29': [facet_a2], '30': [facet_a2], '31': [facet_a2],
                              '32': [facet_a2], '33': [facet_a2], '34': [facet_a2], '35': [facet_a2], '36': [facet_a2],
                              '37': [facet_a2], '38': [facet_a2], '39': [facet_a2], '40': [facet_a2], '41': [facet_a2],
                              '42': [facet_a2], '43': [facet_a2], '45': [facet_a2], '46': [facet_a2], '47': [facet_a2],
                              '48': [facet_a2], '49': [facet_a2], '50': [facet_a2], '51': [facet_a2], '52': [facet_a2],
                              '53': [facet_a2], '54': [facet_a2], '55': [facet_a2], '56': [facet_a2], '57': [facet_a2],
                              '58': [facet_b]}

        documents_per_facet_value_counts = [1, 55]
        standard_deviation = statistics.stdev(documents_per_facet_value_counts)
        self.assertTrue(standard_deviation > 35)

        discriminating_facets = discriminating_algo.get_discriminating_facets(facets_by_document)
        self.assertEqual(len(discriminating_facets), 0)
