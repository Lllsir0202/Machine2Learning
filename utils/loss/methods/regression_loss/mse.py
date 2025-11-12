import numpy as np

class MSE:
    @staticmethod
    def forward(y_true, y_pred):
        """
        Compute Mean Squared Error loss.
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Computed loss
        """
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def backward(y_true, y_pred):
        """
        Compute the gradient of Mean Squared Error loss.
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Gradient of the loss
        """
        n = y_true.shape[0]
        return (2/n) * (y_pred - y_true)