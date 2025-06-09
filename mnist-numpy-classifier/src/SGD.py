# Optimizer method for stochastic gradient descent (SGD)
from optimizer import Optimizer

class SGD(Optimizer):
    def step(self, gradients):
        for key in self.parameters.keys():
            self.parameters[key] -= self.learning_rate * gradients[key]
        return self.parameters