import torch as t
import numpy as np

from src.model import FakeNewsClassifier


class TestModelSpec:

    model = FakeNewsClassifier(5000, 40, 100, 2, 0.2)

    def test_inference_shape(self):
        n = np.random.randint(1, 100)
        x = t.randint(0, 5000, (n, 25))
        y = self.model(x)
        assert y.shape == (n, 1)

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