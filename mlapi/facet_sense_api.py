import requests
from requests.auth import HTTPBasicAuth
from definitions import Definitions
from pathlib import Path
import json


class FacetSenseApi(object):
    QUERY = "{{\"q\": \"{text}\"}}"
    CREDENTIALS_PATH = Path(Definitions.ROOT_DIR + "/appsettings.json")

    def __init__(self):
        with open(FacetSenseApi.CREDENTIALS_PATH, 'r') as file:
            values = json.load(file)
            self.username = values['facetSenseUsername']
            self.password = values['facetSensePassword']
            self.url = values['facetSenseUrl']

    def get_facet_scores(self, text):
        query = FacetSenseApi.QUERY.format(text=text)
        response = requests.post(self.url, data=query, auth=HTTPBasicAuth(self.username, self.password))
        return response.content
