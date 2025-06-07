import numpy as np
import matplotlib.pyplot as plt

class LogicalLearning:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def plot_draw(self, x_axis, losses):
        plt.plot(x_axis, losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.show()