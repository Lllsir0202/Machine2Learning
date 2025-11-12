import numpy as np

class Relu:
    @staticmethod
    def forward(x):
        """Computes the ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x):
        """Computes the gradient of the ReLU function."""
        grad = np.where(x > 0, 1, 0)
        return grad