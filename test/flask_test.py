from pathlib import Path

from definitions import Definitions
from mlapi import app
from flask import json
import unittest


class FlaskTest(unittest.TestCase):
    FACETS_FILE = Path(Definitions.ROOT_DIR + "/test/test_files/facets.bin")
    CREDENTIALS_PATH = Path(Definitions.ROOT_DIR + "/test/test_files/appsettings.json")

    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True
        app.initialize(
            FlaskTest.FACETS_FILE,
            FlaskTest.CREDENTIALS_PATH
        )

    def test_facets_endpoint(self):
        documents = ['a', 'b', 'c', 'd']
        result = self.app.post('/ML/Analyze', data=json.dumps(documents), content_type='application/json')
        self.assertEqual(200, result.status_code)

    def test_filter_facets_endpoint(self):
        request = {
            'Documents': [
                'uri1', 'uri2', 'uri3'
            ],
            'MustHaveFacets': [
                {'Name': 'name1', 'Values': ['value1']},
                {'Name': 'name2', 'Values': ['value2']}
            ],
            'MustNotHaveFacets': [
                {'Name': 'name3', 'Values': ['value3']},
                {'Name': 'name4', 'Values': ['value4']}
            ]
        }
        result = self.app.post('/ML/Filter/Facets', data=json.dumps(request), content_type='application/json')
        self.assertEqual(200, result.status_code)

    def test_facets_values_endpoint(self):
        request = ['@y', '@m']
        result = self.app.post('/ML/Facets', data=json.dumps(request), content_type='application/json')
        self.assertEqual(200, result.status_code)


if __name__ == '__main__':
    unittest.main()
