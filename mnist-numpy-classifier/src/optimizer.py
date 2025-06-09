# Optimizer class for managing model parameters and learning rate

class Optimizer:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate
