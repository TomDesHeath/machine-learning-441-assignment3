"""
Stochastic Gradient Descent (SGD) with Momentum Optimizer.
Author: Tom Des Heath (24888923)
"""
import numpy as np
import time

class SGDOptimizer:
    """
    Stochastic / Batch Gradient Descent optimizer with momentum acceleration.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, batch_size: int = None):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

    def optimize(self, model, X: np.ndarray, Y: np.ndarray, max_epochs: int = 1000, tol: float = 1e-6):
        w = model.get_params()
        v = np.zeros_like(w)
        loss_history = []
        n_samples = X.shape[0]

        start_time = time.time()
        func_evals = 0

        for epoch in range(max_epochs):
            if self.batch_size is None or self.batch_size >= n_samples:
                # Full-batch
                loss, grad = model.compute_loss_and_grad(w, X, Y)
                func_evals += 1
                
                v = self.momentum * v - self.learning_rate * grad
                w = w + v
            else:
                # Mini-batch SGD
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]
                
                epoch_loss = 0.0
                num_batches = int(np.ceil(n_samples / self.batch_size))
                
                for b in range(num_batches):
                    start_idx = b * self.batch_size
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    X_b = X_shuffled[start_idx:end_idx]
                    Y_b = Y_shuffled[start_idx:end_idx]
                    
                    b_loss, grad = model.compute_loss_and_grad(w, X_b, Y_b)
                    func_evals += 1
                    epoch_loss += b_loss * (end_idx - start_idx)
                    
                    v = self.momentum * v - self.learning_rate * grad
                    w = w + v
                
                loss = epoch_loss / n_samples

            model.set_params(w)
            loss_history.append(loss)

            if loss < tol or (len(loss_history) > 10 and abs(loss_history[-1] - loss_history[-2]) < 1e-8):
                break

        elapsed_time = time.time() - start_time
        return {
            "w_opt": w,
            "loss_history": loss_history,
            "iterations": len(loss_history),
            "func_evals": func_evals,
            "time": elapsed_time
        }
