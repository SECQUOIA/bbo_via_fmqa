"""
Factorization Machine implemented with PyTorch
"""

__all__ = [
    "FactorizationMachine"
]

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def triu_mask(input_size, device="cpu"):
    """Generate a square matrix with upper triangular elements = 1 and others = 0"""
    mask = torch.arange(input_size, device=device).unsqueeze(0)
    return (mask.t() < mask) * 1.0


def VtoQ(V):
    """Calculate interaction strength by inner product of feature vectors.

    Args:
        V: Tensor of shape (k, d)
    Returns:
        Q: Tensor of shape (d, d)
    """
    Q = torch.mm(V.t(), V)   # (d, d)
    return Q * triu_mask(Q.shape[0], device=V.device)


class QuadraticLayer(nn.Module):
    """A neural network layer which applies quadratic function on the input.

    This class defines train() method for easy use.
    """

    def __init__(self):
        super().__init__()
        self.trainer = None

    def train_model(self, x, y, num_epoch=100, learning_rate=1.0e-2):
        """Training loop with Adam optimizer"""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Ensure tensors
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)

        for epoch in range(num_epoch):
            optimizer.zero_grad()
            output = self(x).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

   
    def train(self, x=None, y=None, num_epoch=100, learning_rate=1e-2):
        """Training of the regression model using Adam optimizer.
        """
        if x is None and y is None:
            # No args. behave like PyTorch’s default .train()
            return super().train(True)
        else:
            return self.train_model(x, y, num_epoch, learning_rate)



class FactorizationMachine(QuadraticLayer):
    def __init__(self, input_size, factorization_size=8, act="identity"):
        super().__init__()
        self.input_size = input_size
        self.factorization_size = factorization_size

        # Activation
        self.act = {
            "identity": lambda x: x,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh
        }[act]
        
        # Parameters
        self.h = nn.Parameter(torch.randn(input_size))       # linear weights
        self.bias = nn.Parameter(torch.zeros(1))             # bias

        if factorization_size > 0:
            self.V = nn.Parameter(torch.randn(factorization_size, input_size))
        else:
            self.V = nn.Parameter(torch.zeros(1, input_size))  # dummy V



    def init_params(self, mean=0.0, std=0.01):
        """Reinitialize all learnable parameters"""
        nn.init.normal_(self.h, mean=mean, std=std)
        nn.init.normal_(self.V, mean=mean, std=std)
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        """Forward pass

        Args:
            x: Tensor of shape (N, d)
        """
        linear_term = self.bias + torch.matmul(x, self.h)  # (N,)
        if self.factorization_size <= 0:
            return self.act(linear_term)

        Q = VtoQ(self.V)  # (d, d)
        Qx = torch.matmul(x, Q)  # (N, d)
        quadratic_term = torch.sum(x * Qx, dim=1)  # (N,)

        return self.act(linear_term + quadratic_term)

    def get_bhQ(self):
        """Returns bias, linear weights, and Q matrix"""
        with torch.no_grad():
            Q = VtoQ(self.V if self.factorization_size > 0 else torch.zeros_like(self.V))
        return self.bias.item(), self.h.detach().cpu().numpy(), Q.detach().cpu().numpy()
