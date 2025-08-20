import torch
import torch.nn as nn
from .base_model import BaseModel

class FNN(BaseModel):
    def __init__(self, input_size, output_size, hidden_size=[10,10]):
        super().__init__(input_size, output_size)
        self.hidden_size = hidden_size
        self.build()

    def build(self):
        layers = []
        in_size = self.input_size

        # Add hidden layers
        for h_size in self.hidden_size:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size

        # Add output layer
        layers.append(nn.Linear(in_size, self.output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    # Implement abstract methods minimally
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        pass

    def predict(self, X):
        with torch.no_grad():
            return self.forward(X)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.build()
        self.load_state_dict(torch.load(path))
