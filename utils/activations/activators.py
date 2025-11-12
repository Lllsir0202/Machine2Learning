import numpy as np
from activation_functions import *

class Activators:
    def __init__(self, activation_type: str):
        self.activation_type = activation_type

    # TODO: Add more activation functions as needed
    def activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type.lower() == 'sigmoid':
            return Sigmoid.forward(x)
        elif self.activation_type.lower() == 'tanh':
            return Tanh.forward(x)
        elif self.activation_type.lower() == 'relu':
            return Relu.forward(x)
        elif self.activation_type.lower() == 'leaky_relu':
            return Leaky_Relu.forward(x)
        elif self.activation_type.lower() == 'softmax':
            return Softmax.forward(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_type}")
        
    def deactivate(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type.lower() == 'sigmoid':
            return Sigmoid.backward(x)
        elif self.activation_type.lower() == 'tanh':
            return Tanh.backward(x)
        elif self.activation_type.lower() == 'relu':
            return Relu.backward(x)
        elif self.activation_type.lower() == 'leaky_relu':
            return Leaky_Relu.backward(x)
        elif self.activation_type.lower() == 'softmax':
            return ValueError("Softmax does not have a standard backward function.")
        else:
            raise ValueError(f"Unknown activation function: {self.activation_type}")