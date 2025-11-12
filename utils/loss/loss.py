import numpy as np
from loss_functions import *

class Loss:
    LOSS_MAP = {
    }

    def __init__(self, loss_type: str):
        self.loss_type = loss_type.lower()
        if self.loss_type not in self.LOSS_MAP:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        self.loss_class = self.LOSS_MAP[self.loss_type]

    