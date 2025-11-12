import numpy as np

class Leaky_Relu:
    @staticmethod
    def forward(x, alpha: float = 0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def backward(x, alpha: float = 0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx