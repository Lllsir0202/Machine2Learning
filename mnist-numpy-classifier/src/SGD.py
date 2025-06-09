# Optimizer method for stochastic gradient descent (SGD)
from optimizer import Optimizer

class SGD(Optimizer):
    def step(self, parameters, gradients):
        for key in parameters.keys():
            parameters[key] -= self.learning_rate * gradients[key]
        return parameters