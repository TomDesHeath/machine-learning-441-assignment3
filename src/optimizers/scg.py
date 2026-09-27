"""
Scaled Conjugate Gradient (SCG) Optimizer (Møller 1993).
Supervised neural network training without line search.
Author: Tom Des Heath (24888923)
"""
import numpy as np
import time

class SCGOptimizer:
    """
    Scaled Conjugate Gradient optimizer according to Møller (1993).
    """
    def __init__(self, sigma: float = 1e-4, lambda_init: float = 1e-6):
        self.sigma_val = sigma
        self.lambda_init = lambda_init

    def optimize(self, model, X: np.ndarray, Y: np.ndarray, max_epochs: int = 1000, tol: float = 1e-6):
        w = model.get_params()
        num_params = len(w)

        # Initial loss and gradient
        E_k, g_k = model.compute_loss_and_grad(w, X, Y)
        r_k = -g_k
        p_k = r_k.copy()
        
        lambda_k = self.lambda_init
        lambda_bar_k = 0.0
        success = True

        loss_history = [E_k]
        func_evals = 1
        start_time = time.time()

        for k in range(1, max_epochs + 1):
            p_norm = np.linalg.norm(p_k)
            if p_norm == 0.0:
                break

            if success:
                # Step 3: Calculate second-order information via finite differences
                sigma_k = self.sigma_val / p_norm
                w_plus = w + sigma_k * p_k
                _, g_plus = model.compute_loss_and_grad(w_plus, X, Y)
                func_evals += 1
                
                s_k = (g_plus - g_k) / sigma_k
                delta_k = np.dot(p_k, s_k)

            # Step 4: Scale Hessian approximation
            delta_k += (lambda_k - lambda_bar_k) * (p_norm ** 2)

            # Step 5: Make Hessian positive definite if required
            if delta_k <= 0:
                lambda_bar_k = 2.0 * (lambda_k - delta_k / (p_norm ** 2))
                delta_k = -delta_k + lambda_k * (p_norm ** 2)
                lambda_k = lambda_bar_k

            # Step 6: Calculate step size alpha_k
            mu_k = np.dot(p_k, r_k)
            if delta_k == 0.0:
                break
            alpha_k = mu_k / delta_k

            # Step 7: Calculate comparison parameter Delta_k
            w_new = w + alpha_k * p_k
            E_new, g_new = model.compute_loss_and_grad(w_new, X, Y)
            func_evals += 1

            Delta_k = 2.0 * delta_k * (E_k - E_new) / (mu_k ** 2) if mu_k != 0 else 0.0

            # Step 8: Evaluate success condition
            if Delta_k >= 0:
                w = w_new
                E_k = E_new
                r_old = r_k.copy()
                g_k = g_new
                r_k = -g_k
                lambda_bar_k = 0.0
                success = True

                # Restart direction check
                if k % num_params == 0:
                    p_k = r_k.copy()
                else:
                    beta_k = (np.dot(r_k, r_k) - np.dot(r_k, r_old)) / mu_k if mu_k != 0 else 0.0
                    p_k = r_k + beta_k * p_k

                if Delta_k >= 0.75:
                    lambda_k *= 0.25
            else:
                lambda_bar_k = lambda_k
                success = False

            if Delta_k < 0.25:
                lambda_k += (delta_k * (1.0 - Delta_k)) / (p_norm ** 2)

            model.set_params(w)
            loss_history.append(E_k)

            # Convergence check
            if np.linalg.norm(r_k) < tol or E_k < tol:
                break

        elapsed_time = time.time() - start_time
        return {
            "w_opt": w,
            "loss_history": loss_history,
            "iterations": len(loss_history),
            "func_evals": func_evals,
            "time": elapsed_time
        }
