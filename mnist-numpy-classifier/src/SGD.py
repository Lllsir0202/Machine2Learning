import numpy as np
from optimizer import Optimizer

class SGD(Optimizer):
    def step(self, gradients):
        for key in self.parameters.keys():
            self.parameters[key] -= self.learning_rate * gradients[key]
        return self.parameters