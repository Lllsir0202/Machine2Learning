import numpy as np

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