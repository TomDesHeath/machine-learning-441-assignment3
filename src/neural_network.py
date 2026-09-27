"""
Feedforward Neural Network (FNN) Architecture with Analytic Backpropagation.
Supports Regression (MSE) and Multi-Class Classification (Softmax + Cross Entropy).
Author: Tom Des Heath (24888923)
"""
import numpy as np

class FeedforwardNeuralNetwork:
    """
    Single hidden layer feedforward neural network with vectorized forward/backward passes.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        problem_type: str = "classification",
        activation: str = "sigmoid",
        gamma: float = 1e-4,
        seed: int = 42
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.problem_type = problem_type
        self.activation = activation
        self.gamma = gamma
        self.seed = seed

        # Glorot / Xavier Uniform Initialization
        rng = np.random.RandomState(seed)
        limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = rng.uniform(-limit1, limit1, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)

        limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = rng.uniform(-limit2, limit2, (output_dim, hidden_dim))
        self.b2 = np.zeros(output_dim)

    def _hidden_activation(self, a: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            # Numerically stable sigmoid
            a_clipped = np.clip(a, -500.0, 500.0)
            return np.where(a_clipped >= 0, 1.0 / (1.0 + np.exp(-a_clipped)), np.exp(a_clipped) / (1.0 + np.exp(a_clipped)))
        elif self.activation == "tanh":
            return np.tanh(a)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _hidden_activation_deriv(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return y * (1.0 - y)
        elif self.activation == "tanh":
            return 1.0 - y ** 2
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _output_activation(self, a: np.ndarray) -> np.ndarray:
        if self.problem_type == "regression":
            return a
        elif self.problem_type == "classification":
            # Stable Softmax
            exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
            return exp_a / np.sum(exp_a, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    def get_params(self) -> np.ndarray:
        """Flatten model weights into a 1D vector."""
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel()
        ])

    def set_params(self, w: np.ndarray):
        """Unpack 1D parameter vector into layer weights and biases."""
        idx1 = self.hidden_dim * self.input_dim
        idx2 = idx1 + self.hidden_dim
        idx3 = idx2 + self.output_dim * self.hidden_dim

        self.W1 = w[:idx1].reshape(self.hidden_dim, self.input_dim)
        self.b1 = w[idx1:idx2]
        self.W2 = w[idx2:idx3].reshape(self.output_dim, self.hidden_dim)
        self.b2 = w[idx3:]

    def forward(self, X: np.ndarray, w: np.ndarray = None) -> tuple:
        """Forward pass given input X and optional parameter vector w."""
        if w is not None:
            self.set_params(w)

        a1 = np.dot(X, self.W1.T) + self.b1
        y1 = self._hidden_activation(a1)
        a2 = np.dot(y1, self.W2.T) + self.b2
        y2 = self._output_activation(a2)

        return a1, y1, a2, y2

    def compute_loss_and_grad(self, w: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple:
        """
        Compute total loss E(w) and exact analytical gradient g(w) = grad E(w).
        """
        N = X.shape[0]
        a1, y1, a2, y2 = self.forward(X, w=w)

        # Regularization loss
        l2_reg = 0.5 * self.gamma * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))

        if self.problem_type == "regression":
            # MSE loss: (1 / (2N)) * sum ||y2 - Y||^2
            loss = 0.5 * np.mean(np.sum((y2 - Y) ** 2, axis=1)) + l2_reg
            delta2 = (y2 - Y) / N
        else:
            # Cross-Entropy loss: -(1/N) * sum Y * log(y2)
            eps = 1e-15
            y2_clipped = np.clip(y2, eps, 1.0 - eps)
            loss = -np.mean(np.sum(Y * np.log(y2_clipped), axis=1)) + l2_reg
            delta2 = (y2 - Y) / N

        # Hidden layer delta
        da1 = self._hidden_activation_deriv(a1, y1)
        delta1 = np.dot(delta2, self.W2) * da1

        # Parameter gradients
        dW2 = np.dot(delta2.T, y1) + self.gamma * self.W2
        db2 = np.sum(delta2, axis=0)
        dW1 = np.dot(delta1.T, X) + self.gamma * self.W1
        db1 = np.sum(delta1, axis=0)

        grad = np.concatenate([
            dW1.ravel(),
            db1.ravel(),
            dW2.ravel(),
            db2.ravel()
        ])

        return loss, grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input matrix X."""
        _, _, _, y2 = self.forward(X)
        if self.problem_type == "classification":
            return np.argmax(y2, axis=1)
        else:
            return y2

    def check_gradients(self, X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
        """Verify analytical gradients against finite differences."""
        w = self.get_params()
        _, grad_analytic = self.compute_loss_and_grad(w, X, Y)
        grad_num = np.zeros_like(w)

        for i in range(len(w)):
            w_plus = w.copy()
            w_plus[i] += eps
            loss_plus, _ = self.compute_loss_and_grad(w_plus, X, Y)

            w_minus = w.copy()
            w_minus[i] -= eps
            loss_minus, _ = self.compute_loss_and_grad(w_minus, X, Y)

            grad_num[i] = (loss_plus - loss_minus) / (2.0 * eps)

        rel_error = np.linalg.norm(grad_analytic - grad_num) / (np.linalg.norm(grad_analytic) + np.linalg.norm(grad_num) + 1e-12)
        return float(rel_error)
