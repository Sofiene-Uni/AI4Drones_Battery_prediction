import torch
import torch.nn as nn
from .base_model import BaseModel

class RNN(BaseModel):
    def build(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.params.get("hidden_size", 64)),
            nn.ReLU(),
            nn.Linear(self.params.get("hidden_size", 64), self.output_size)
        )

    def train(self, X_train, y_train, X_val, y_val):
        # basic training loop
        pass

    def predict(self, X):
        with torch.no_grad():
            return self.model(X)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.build()
        self.model.load_state_dict(torch.load(path))
