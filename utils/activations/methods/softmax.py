import numpy as np

class Softmax:
    @staticmethod
    def forward(x):
        """Compute the softmax of each row of the input x."""
        return np.exp(x) / np.sum(np.exp(x))