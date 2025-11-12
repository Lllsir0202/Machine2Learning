import numpy as np

class MAE:
    """
    Mean Absolute Error (MAE) Loss Function for regression tasks.

    formula:
        loss = (1/n) * Σ|y_true - y_pred|
        gradient = (1/n) * sign(y_pred - y_true)
    """
    @staticmethod
    def forward(y_true, y_pred):
        """
        Compute Mean Absolute Error loss.
        
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Computed loss
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def backward(y_true, y_pred):
        """
        Compute the gradient of Mean Absolute Error loss.
        
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Gradient of the loss
        """
        n = y_true.shape[0]
        grad = np.sign(y_pred - y_true)
        return grad / n