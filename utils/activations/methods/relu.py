import numpy as np

"""
ReLU Activation Function Module
formula:
    forward: relu(x) = max(0, x)
    backward: 1 if x > 0 else 0
"""
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