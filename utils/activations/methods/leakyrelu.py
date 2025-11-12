import numpy as np

"""
Leaky ReLU Activation Function Module
formula:
    forward: leaky_relu(x) = x if x > 0 else alpha * x
    backward: 1 if x > 0 else alpha
(Compared with ReLU, Leaky ReLU allows a small, non-zero gradient when the unit is not active.)
"""

class Leaky_Relu:
    @staticmethod
    def forward(x, alpha: float = 0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def backward(x, alpha: float = 0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx