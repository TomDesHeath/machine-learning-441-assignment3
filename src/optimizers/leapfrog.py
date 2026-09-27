"""
LeapFrog Optimization (LFROG) Algorithm (Snyman 1982, 1983).
Dynamic physical system tracking a particle in an error force field.
Author: Tom Des Heath (24888923)
"""
import numpy as np
import time

class LeapFrogOptimizer:
    """
    LeapFrog optimizer based on Snyman (1982, 1983) dynamic particle trajectory method.
    """
    def __init__(self, delta_t: float = 0.1, dt_min: float = 1e-5, dt_max: float = 0.5):
        self.delta_t_init = delta_t
        self.dt_min = dt_min
        self.dt_max = dt_max

    def optimize(self, model, X: np.ndarray, Y: np.ndarray, max_epochs: int = 1000, tol: float = 1e-6):
        w = model.get_params()
        dt = self.delta_t_init

        # Initial force/acceleration a_0 = -grad E(w_0)
        E_curr, grad_curr = model.compute_loss_and_grad(w, X, Y)
        a_curr = -grad_curr
        
        # Initial half-step velocity
        v_half = 0.5 * a_curr * dt
        
        loss_history = [E_curr]
        func_evals = 1
        start_time = time.time()

        for step in range(1, max_epochs + 1):
            # 1. Update position
            w_next = w + v_half * dt
            
            # 2. Acceleration at new position
            E_next, grad_next = model.compute_loss_and_grad(w_next, X, Y)
            func_evals += 1
            a_next = -grad_next

            # 3. Next half-step velocity
            v_half_next = v_half + a_next * dt

            # 4. Full-step velocity estimation
            v_full = 0.5 * (v_half + v_half_next)

            # 5. Interfering strategy: check if moving uphill (dot product of velocity and acceleration < 0)
            if np.dot(v_full, a_next) < 0.0 or E_next > E_curr:
                # Reset velocity at local minimum / valley floor
                w_next = 0.5 * (w + w_next)
                E_curr, grad_next = model.compute_loss_and_grad(w_next, X, Y)
                a_next = -grad_next
                func_evals += 1
                
                v_half_next = 0.5 * a_next * dt
                dt = max(dt * 0.5, self.dt_min)
            else:
                E_curr = E_next
                # Slightly expand timestep if trajectory is smooth
                dt = min(dt * 1.02, self.dt_max)

            w = w_next
            v_half = v_half_next
            a_curr = a_next

            model.set_params(w)
            loss_history.append(E_curr)

            # Convergence check
            if np.linalg.norm(grad_next) < tol or E_curr < tol:
                break

        elapsed_time = time.time() - start_time
        return {
            "w_opt": w,
            "loss_history": loss_history,
            "iterations": len(loss_history),
            "func_evals": func_evals,
            "time": elapsed_time
        }
