import unittest
from mlapi.model.facet import Facet
from mlapi.question_generator import QuestionGenerator


class TestQuestions(unittest.TestCase):
    def test_when_same_facet_then_return_no_questions(self):
        question_generator = QuestionGenerator()
        facet_a = Facet('NameA', 'ValueA')
        facets_by_document = {'1': [facet_a], '2': [facet_a]}

        questions = question_generator.generate_questions(facets_by_document)
        self.assertEqual(len(questions), 0)

    def test_no_facets_then_return_no_questions(self):
        question_generator = QuestionGenerator()
        questions = question_generator.generate_questions({})
        self.assertEqual(len(questions), 0)

    def test_one_discriminating_facet_then_return_one_element_question(self):
        question_generator = QuestionGenerator()
        facet_a1 = Facet('NameA', 'ValueA1')
        facet_a2 = Facet('NameA', 'ValueA2')
        facets_by_document = {'1': [facet_a1], '2': [facet_a2], '3': [facet_a2]}

        questions = question_generator.generate_questions(facets_by_document)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].facet_name, "NameA")
        self.assertEqual(sorted(questions[0].facet_values), ['ValueA1', 'ValueA2'])

    def test_similar_values_different_facets_then_return_one_question(self):
        question_generator = QuestionGenerator()
        facet_a1 = Facet('NameA', 'ValueA')
        facet_a2 = Facet('NameA', 'ValueA')
        facet_a3 = Facet('NameA', 'ValueA')
        facet_b1 = Facet('NameB', 'ValueB1')
        facet_b2 = Facet('NameB', 'ValueB2')
        facets_by_document = {'1': [facet_a1, facet_b1], '2': [facet_a2, facet_b1], '3': [facet_a3, facet_b2]}

        questions = question_generator.generate_questions(facets_by_document)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].facet_name, "NameB")
        self.assertEqual(sorted(questions[0].facet_values), ['ValueB1', 'ValueB2'])

    def test_similar_values_different_facets_then_return_no_question(self):
        question_generator = QuestionGenerator()
        facet_a1 = Facet('NameA', 'ValueA')
        facet_a2 = Facet('NameA', 'ValueA')
        facets_by_document = {'1': [facet_a1], '2': [facet_a2], '3': [facet_a2]}

        questions = question_generator.generate_questions(facets_by_document)
        self.assertEqual(len(questions), 0)

    def test_multiple_values_then_return_multiple_questions(self):
        question_generator = QuestionGenerator()
        facet_a = Facet('NameA', 'ValueA')
        facet_b = Facet('NameA', 'ValueB')
        facet_c = Facet('NameA', 'ValueC')
        facet_d = Facet('NameB', 'Value1')
        facet_e = Facet('NameB', 'Value2')
        facet_f = Facet('NameB', 'Value3')
        facet_g = Facet('NameB', 'Value4')
        facets_by_document = {'1': [facet_a, facet_e, facet_c, facet_g], '2': [facet_b, facet_d, facet_f, facet_g], '3': [facet_a, facet_g, facet_f]}

        questions = sorted(question_generator.generate_questions(facets_by_document), key=lambda x: x.facet_name)
        self.assertEqual(questions[0].facet_name, 'NameA')
        self.assertEqual(sorted(questions[0].facet_values), ['ValueA', 'ValueB', 'ValueC'])
        self.assertEqual(questions[1].facet_name, 'NameB')
        self.assertEqual(sorted(questions[1].facet_values), ['Value1', 'Value2', 'Value3'])