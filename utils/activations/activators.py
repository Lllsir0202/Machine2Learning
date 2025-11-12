import numpy as np
from activation_functions import *

class Activators:
    ACTIVATION_MAP = {
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'relu': Relu,
        'leaky_relu': Leaky_Relu,
        'softmax': Softmax
    }

    def __init__(self, activation_type: str):
        self.activation_type = activation_type.lower()
        if self.activation_type not in self.ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation function: {activation_type}")
        self.activation_class = self.ACTIVATION_MAP[self.activation_type]

    # TODO: Add more activation functions as needed
    def activate(self, x: np.ndarray) -> np.ndarray:
        return self.activation_class.forward(x)
        
    def deactivate(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.activation_class, 'backward'):
            raise NotImplementedError(f"Backward method not implemented for {self.activation_type}")
        
        return self.activation_class.backward(x)