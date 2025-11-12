import numpy as np

"""
Softmax Activation Function Module

formula:
    forward: softmax(x_i) = exp(x_i) / Σ(exp(x_j))
"""

class Softmax:
    @staticmethod
    def forward(x):
        """Compute the softmax of each row of the input x."""
        return np.exp(x) / np.sum(np.exp(x))