from mlapi import app
from flask import json
import unittest


class FlaskTest(unittest.TestCase):

    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    def test_facets_endpoint(self):
        documents = ['dfsdf', 'sdfsdf','sdfsdf','fsdfsdf']
        result = self.app.post('/ML/Analyze', data=json.dumps(documents), content_type='application/json')
        self.assertEqual(result.status_code, 200)

    def test_filter_facets_endpoint(self):
        request = {'Documents': ['uri1', 'uri2', 'uri3'], 'MustHaveFacets': [{'Name': 'name1', 'Value': 'value1'}, {'Name': 'name2', 'Value': 'value2'}], 'MustNotHaveFacets': [{'Name': 'name3', 'Value': 'value3'}, {'Name': 'name4', 'Value': 'value4'}]}
        result = self.app.post('/ML/Filter/Facets', data=json.dumps(request), content_type='application/json')
        self.assertEqual(result.status_code, 200)

    def test_facets_values_endpoint(self):
        request = ['@y', '@m']
        result = self.app.post('/ML/Facets', data=json.dumps(request), content_type='application/json')
        self.assertEqual(result.status_code, 200)


if __name__ == '__main__':
    unittest.main()