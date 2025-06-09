import numpy as np

from activations import Activations
from losses import cross_entropy, softmax

class Model:
    def __init__(self, input_size, output_size, activation='relu'):
        # First layer weights and biases
        # Use He initialization for ReLU activation
        self.activation = activation
        self.w1 = np.random.randn(input_size, 128) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((128,))
        # Second layer weights and biases
        self.w2 = np.random.randn(128, output_size) * np.sqrt(2. / 128)
        self.b2 = np.zeros((output_size,))


    def forward(self, x):
        z1 = x @ self.w1 + self.b1

        activation1 = Activations(self.activation)
        a1 = activation1.activate(z1)

        z2 = a1 @ self.w2 + self.b2
        prob = softmax(z2)

        # a2 is the output probabilities

        self.z1 = z1
        self.a1 = a1
        self.z2 = z2
        self.prob = prob

        # Return prob
        return prob



    def backward(self, x, y, learning_rate = 0.01):
        # Compute gradients for the second layer
        delta2 = self.prob - y

        dw2 = self.a1.T @ delta2 # Shape: (128, output_size)
        db2 = np.sum(delta2, axis=0) # Shape: (output_size,)
        # Compute gradients for the first layer

        activation = Activations('relu')
        delta1 = (delta2 @ self.w2.T) * activation.derivative(self.z1)
        dw1 = x.T @ delta1
        db1 = np.sum(delta1, axis=0)

        # Update weights and biases
        self.dw1 = dw1
        self.db1 = db1
        self.dw2 = dw2
        self.db2 = db2

        return