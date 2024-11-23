# src/evaluator.py

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, test_generator, steps):
        scores = self.model.evaluate(test_generator, steps=steps)
        return scores
