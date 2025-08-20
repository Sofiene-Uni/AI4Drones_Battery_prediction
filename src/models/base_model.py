import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, input_size, output_size, **kwargs):
        """
        Base class for all models.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            kwargs: Extra parameters like hidden_size, num_layers, etc.
        """
        super(BaseModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.params = kwargs  # store extra parameters

    @abstractmethod
    def build(self):
        """Define architecture"""
        pass

    @abstractmethod
    def train_model(self, X_train, y_train, X_val=None, y_val=None, **train_kwargs):
        """Training loop"""
        pass

    @abstractmethod
    def predict(self, X):
        """Return predictions"""
        pass

    @abstractmethod
    def save(self, path):
        """Save trained model"""
        pass

    @abstractmethod
    def load(self, path):
        """Load trained model"""
        pass
