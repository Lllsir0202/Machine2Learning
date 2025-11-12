import numpy as np

"""
Tanh Activation Function Module

formula:
    forward: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    backward: (1 - tanh²(x))
"""

class Tanh:
    @staticmethod
    def forward(x):
        """Compute the tanh activation function."""
        return np.tanh(x)

    @staticmethod
    def backward(x):
        """Compute the derivative of the tanh function."""
        tanh_x = Tanh.forward(x)
        return 1 - tanh_x ** 2