class FacetValueDto(object):
    def __init__(self, value, score):
        self.value = value
        self.score = score

    def to_dict(self):
        return {"value": self.value, "score": self.score}
