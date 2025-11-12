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
        """Optimization: Subtract the max for numerical stability"""
        stable_x = x - np.max(x)
        exp_x = np.exp(stable_x)
        return exp_x / np.sum(exp_x)