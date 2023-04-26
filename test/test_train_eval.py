import torch as t
import numpy as np

from src.model import FakeNewsClassifier


class TestTrainEval:

    model = FakeNewsClassifier(5000, 40, 100, 2, 0.2)

    def test_nondet_train(self):
        x = t.randint(0, 5000, (10, 25))

        self.model.train()
        y_1 = self.model(x)
        y_2 = self.model(x)
        assert y_1.shape == y_2.shape
        assert not t.equal(y_1, y_2)

    def test_det_eval(self):
        x = t.randint(0, 5000, (10, 25))

        self.model.eval()
        y_1 = self.model(x)
        y_2 = self.model(x)
        assert y_1.shape == y_2.shape
        assert t.equal(y_1, y_2)