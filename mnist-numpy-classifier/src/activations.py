import numpy as np

class Activations:
    def __init__(self, activation_type: str):
        self.activation_type = activation_type

    def activate(self, x):
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
    
    # Input is the INPUT to the activation function
    def derivative(self, x):
        if self.activation_type == 'sigmoid':
            sig = self.activate(x)
            return sig * (1 - sig)
        elif self.activation_type == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_type == 'relu':
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError(f"Unsupported derivation type: {self.activation_type}")