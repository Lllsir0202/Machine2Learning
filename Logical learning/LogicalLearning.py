import numpy as np
import matplotlib.pyplot as plt

class LogicalLearning:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - x ** 2
    
    def plot_draw(self, x_axis, losses):
        plt.plot(x_axis, losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.show()