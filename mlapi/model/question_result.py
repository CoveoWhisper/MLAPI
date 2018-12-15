
class QuestionResult(object):

    def __init__(self, question, score):
        self.question = question
        self.score = score

    def to_dict(self):
        return {"facetQuestion": self.question,
                "score": self.score}
