from json import JSONEncoder


class FacetValues(JSONEncoder):

    def __init__(self, name, values, score):
        self.name = name
        self.values = values
        self.score = score

    def to_dict(self):
        return {"_type": FacetValues.__name__, "name": self.name, "values": self.values, "score": self.score}
