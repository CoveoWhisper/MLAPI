from json import JSONEncoder


class FacetValues(JSONEncoder):

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def to_dict(self):
        return {"_type": FacetValues.__name__, "name": self.name, "values": self.values}
