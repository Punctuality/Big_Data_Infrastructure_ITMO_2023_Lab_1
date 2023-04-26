import torch as t
import numpy as np

from src.model import FakeNewsClassifier


class TestShapes:

    model = FakeNewsClassifier(5000, 40, 100, 2, 0.2)

    def test_inference_shape(self):
        n = np.random.randint(1, 100)
        x = t.randint(0, 5000, (n, 25))
        y = self.model(x)
        assert y.shape == (n, 1)