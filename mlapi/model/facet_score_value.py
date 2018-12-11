from mlapi.model.facet_values import FacetValues


class FacetScoreValues(FacetValues):
    def __init__(self, name, values, score):
        super().__init__(name,values)
        self.score = score

    def to_dict(self):
        return super().to_dict().update({"score": self.score})
