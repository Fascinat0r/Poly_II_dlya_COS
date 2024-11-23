# evaluator.py

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        return scores  # возвращает loss и accuracy
