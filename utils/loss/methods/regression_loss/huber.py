import numpy as np

"""
Huber Loss Function for regression tasks.

formula:
    loss = (1/n) * Σ Lδ(y_true - y_pred)
    gradient = (1/n) * δ * sign(y_pred - y_true)
    where Lδ is the Huber loss function defined as:
        Lδ(x) = 0.5 * x^2               if |x| <= δ
               = δ * (|x| - 0.5 * δ)   if |x| > δ
"""

class Huber:
    @staticmethod
    def forward(y_true, y_pred, delta=1.0):
        """
        Compute Huber loss.
        
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param delta: Threshold parameter for Huber loss
        :return: Computed loss
        """
        error = y_true - y_pred
        if np.abs(error) <= delta:
            return 0.5 * np.mean(np.square(error))
        else:
            return delta * (np.mean(np.abs(error)) - 0.5 * delta)
    
    @staticmethod
    def backward(y_true, y_pred, delta=1.0):
        """
        Compute the gradient of Huber loss.
        
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param delta: Threshold parameter for Huber loss
        :return: Gradient of the loss
        """
        error = y_pred - y_true
        is_small_error = np.abs(error) <= delta
        grad = np.where(is_small_error, error, delta * np.sign(error))
        n = y_true.shape[0]
        return grad / n