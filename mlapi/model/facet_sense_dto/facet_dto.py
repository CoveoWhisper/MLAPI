class FacetDto(object):
    def __init__(self, name, score, values):
        self.name = name
        self.score = score
        self.values = values

    def to_dict(self):
        return {"facetName": self.name, "score": self.score, "values": self.values}
