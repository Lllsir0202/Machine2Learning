import numpy as np

"""
Sigmoid Activation Function Module
formula:
    forward: sigmoid(x) = 1 / (1 + exp(-x))
    backward: sigmoid(x) * (1 - sigmoid(x))
"""

class Sigmoid:
    @staticmethod
    def forward(x):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        """Compute the derivative of the sigmoid function."""
        sig = Sigmoid.forward(x)
        return sig * (1 - sig)