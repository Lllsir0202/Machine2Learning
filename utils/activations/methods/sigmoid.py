import numpy as np

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