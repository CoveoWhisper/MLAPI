class Question(object):

    def __init__(self, id, facet_name, facet_values, answer, status):
        self.id = id
        self.facet_name = facet_name
        self.facet_values = facet_values
        self.answer = answer
        self.status = status

    def to_dict(self):
        return {"id": self.id,
                "facetName": self.facet_name,
                "facetValues": self.facet_values,
                "answer": self.answer,
                "status": self.status}
