"""
Trainable Binary Quadratic Model based on Factorization Machine (FMBQM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.vartypes import Vartype
from dimod import BQM  # kept for compatibility if used elsewhere

from .factorization_machine import FactorizationMachine

__all__ = [
    "FactorizationMachineBinaryQuadraticModel",
    "FMBQM",
]


class FactorizationMachineBinaryQuadraticModel(BinaryQuadraticModel):
    """FMBQM: Trainable BQM based on Factorization Machine

    Args:
        input_size (int):
            The dimension of input vector.
        vartype (dimod.vartypes.Vartype):
            The type of input vector (BINARY or SPIN).
        act (string, optional):
            Name of activation function applied on FM output:
            "identity", "sigmoid", or "tanh".
    """

    def __init__(self, input_size, vartype, act="identity", **kwargs):
        # Initialization of BQM
        init_linear = {i: 0.0 for i in range(input_size)}
        init_quadratic = {}
        init_offset = 0.0
        super().__init__(init_linear, init_quadratic, init_offset, vartype, **kwargs)

        # PyTorch FM model
        self.fm = FactorizationMachine(input_size, act=act, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_qubo(self):
        return self._fm_to_qubo()

    def to_ising(self):
        return self._fm_to_ising()

    @classmethod
    def from_data(cls, x, y, act="identity",
                  num_epoch=1000, learning_rate=1.0e-2, **kwargs):
        """Create a binary quadratic model by FM regression model
        trained on the provided data.

        Args:
            x (ndarray, int):
                Input vectors of SPIN/BINARY.
            y (ndarray, float):
                Target values.
            act (string, optional):
                Name of activation function applied on FM output:
                "identity", "sigmoid", or "tanh".
            num_epoch (int, optional):
                The number of epoch for training FM model.
            learning_rate (float, optional):
                Learning rate for FM's optimizer.
            **kwargs:

        Returns:
            :class:`.FactorizationMachineBinaryQuadraticModel`
        """
        x = np.asarray(x)
        if np.all((x == 0) | (x == 1)):
            vartype = Vartype.BINARY
        elif np.all((x == -1) | (x == 1)):
            vartype = Vartype.SPIN
        else:
            raise ValueError("input data should be BINARY or SPIN vectors")

        input_size = x.shape[-1]
        fmbqm = cls(input_size, vartype, act, **kwargs)
        fmbqm.train(x, y, num_epoch, learning_rate, init=True)
        return fmbqm

    def train(self, x, y, num_epoch=1000, learning_rate=1.0e-2, init=False):
        """Train FM regression model on the provided data.

        Args:
            x (ndarray, int):
                Input vectors of SPIN/BINARY.
            y (ndarray, float):
                Target values.
            num_epoch (int, optional):
                The number of epoch for training FM model.
            learning_rate (float, optional):
                Learning rate for FM's optimizer.
            init (bool, optional):
                Initialize or not before training.
        """
        x = np.asarray(x)
        y = np.asarray(y, dtype=np.float32)

        if init:
            self.fm.init_params()

        self._check_vartype(x)

        # Delegate actual training to FactorizationMachine (PyTorch)
        self.fm.train(x, y, num_epoch, learning_rate)

        # After training, sync FM parameters into BQM
        if self.vartype == Vartype.SPIN:
            h, J, b = self._fm_to_ising()
            self.offset = b
            for i in range(self.fm.input_size):
                self.linear[i] = h[i]
            for i in range(self.fm.input_size):
                for j in range(i + 1, self.fm.input_size):
                    self.quadratic[(i, j)] = J.get((i, j), 0.0)

        elif self.vartype == Vartype.BINARY:
            Q, b = self._fm_to_qubo()
            self.offset = b
            for i in range(self.fm.input_size):
                self.linear[i] = Q[(i, i)]
            for i in range(self.fm.input_size):
                for j in range(i + 1, self.fm.input_size):
                    self.quadratic[(i, j)] = Q.get((i, j), 0.0)

    def predict(self, x):
        """Predict target value by trained model.

        Args:
            x (ndarray, int or list):
                Input vectors of SPIN/BINARY.

        Returns:
            :obj:`numpy.ndarray`: Predicted values.
        """
        x = np.asarray(x)
        self._check_vartype(x)

        if x.ndim == 1:
            x = x[None, :]

        # Convert to PyTorch tensor
        x_t = torch.from_numpy(x.astype(np.float32))

        # Forward pass without grad
        self.fm.eval()
        with torch.no_grad():
            y_t = self.fm(x_t)

        # Return numpy array
        return y_t.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_vartype(self, x):
        """Verify that input matches this model's vartype."""
        x = np.asarray(x)
        if self.vartype == Vartype.BINARY:
            if np.all((x == 0) | (x == 1)):
                return
        elif self.vartype == Vartype.SPIN:
            if np.all((x == -1) | (x == 1)):
                return
        raise ValueError("input data should be of type", self.vartype)

    def _fm_to_ising(self, scaling=True):
        """Convert trained model into Ising parameters.

        Args:
            scaling (bool, optional):
                Flag for automatic scaling.

        Returns:
            (h_dict, J_dict, offset)
        """
        b, h, J = self.fm.get_bhQ()  # expect numpy arrays

        # Convert from BINARY to SPIN if needed
        if self.vartype is Vartype.BINARY:
            # Standard BINARY→SPIN mapping
            b = b + np.sum(h) / 2.0 + np.sum(J) / 4.0
            h = (2 * h + np.sum(J, axis=0) + np.sum(J, axis=1)) / 4.0
            J = J / 4.0

        if scaling:
            max_h = np.max(np.abs(h)) if h.size > 0 else 0.0
            max_J = np.max(np.abs(J)) if J.size > 0 else 0.0
            scaling_factor = max(max_h, max_J, 1e-12)  # avoid div-by-zero
            b /= scaling_factor
            h /= scaling_factor
            J /= scaling_factor

        # Convert arrays to dicts
        h_dict = {i: float(h[i]) for i in range(len(h))}
        J_dict = {tuple(idx): float(J[idx]) for idx in zip(*J.nonzero())}

        return h_dict, J_dict, float(b)

    def _fm_to_qubo(self, scaling=True):
        """Convert trained model into QUBO parameters.

        Args:
            scaling (bool, optional):
                Flag for automatic scaling.

        Returns:
            (Q_dict, offset)
        """
        b, h, Q = self.fm.get_bhQ()  # expect numpy arrays

        # Convert from SPIN to BINARY if needed
        if self.vartype is Vartype.SPIN:
            # Standard SPIN→BINARY mapping
            b = b - np.sum(h) + np.sum(Q)
            h = 2.0 * (h - np.sum(Q, axis=0) - np.sum(Q, axis=1))
            Q = 4.0 * Q

        # Put h on the diagonal
        Q = np.array(Q, copy=True)
        np.fill_diagonal(Q, h)

        if scaling:
            max_Q = np.max(np.abs(Q)) if Q.size > 0 else 0.0
            scaling_factor = max(max_Q, 1e-12)
            b /= scaling_factor
            Q /= scaling_factor

        # Conversion from full matrix to dict
        Q_dict = {tuple(idx): float(Q[idx]) for idx in zip(*Q.nonzero())}
        # Ensure all diagonals present
        for i in range(Q.shape[0]):
            Q_dict[(i, i)] = float(Q[i, i])

        return Q_dict, float(b)


FMBQM = FactorizationMachineBinaryQuadraticModel

