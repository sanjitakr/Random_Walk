import numpy as np


class GradientDescent:
    def __init__(self, energy_model, step_size=0.01, tol=1e-6, callbacks=None):
        self.energy_model = energy_model
        self.step_size = float(step_size)
        self.tol = float(tol)
        self.callbacks = callbacks or []
    
    def minimize(self, x0, max_iter=10000):
        x = np.asarray(x0, dtype=float).copy()
        trajectory = [x.copy()]
        
        for step in range(max_iter):
            grad = self.energy_model.gradient(x)
            grad_norm = np.linalg.norm(grad)
            if not np.isfinite(grad_norm):
                raise RuntimeError(f"Gradient diverged at step {step}")
            if grad_norm < self.tol:
                break
            x_new = x - self.step_size * grad
            if np.linalg.norm(x_new - x) < self.tol:
                x = x_new
                trajectory.append(x.copy())
                break
            x = x_new
            trajectory.append(x.copy())
            E = self.energy_model.energy(x)
            for cb in self.callbacks:
                cb(step=step, x=x.copy(), energy=E, gradient=grad.copy())
                if hasattr(cb, 'converged') and cb.converged:
                    break
            if any(hasattr(cb, 'converged') and cb.converged for cb in self.callbacks):
                break
        
        return np.array(trajectory)


class ConjugateGradient:
    
    def __init__(self, energy_model, step_size=0.1, tol=1e-6, 
                 callbacks=None, restart_every=None):
        self.energy_model = energy_model
        self.step_size = float(step_size)
        self.tol = float(tol)
        self.callbacks = callbacks or []
        self.restart_every = restart_every
    
    def minimize(self, x0, max_iter=10000):
        x = np.asarray(x0, dtype=float).copy()
        trajectory = [x.copy()]
        g = self.energy_model.gradient(x)
        d = -g
        g_norm2 = np.dot(g, g)
        
        for step in range(max_iter):
            if np.sqrt(g_norm2) < self.tol:
                break
            x_new = x + self.step_size * d
            g_new = self.energy_model.gradient(x_new)
            g_new_norm2 = np.dot(g_new, g_new)
            if not np.isfinite(g_new_norm2):
                raise RuntimeError(f"Gradient diverged at step {step}")
            beta = g_new_norm2 / g_norm2 if g_norm2 > 0 else 0.0
            if self.restart_every and step % self.restart_every == 0:
                beta = 0.0
            d = -g_new + beta * d
            x = x_new
            g = g_new
            g_norm2 = g_new_norm2
            trajectory.append(x.copy())
            E = self.energy_model.energy(x)
            for cb in self.callbacks:
                cb(step=step, x=x.copy(), energy=E, gradient=g.copy())
                
                if hasattr(cb, 'converged') and cb.converged:
                    break
            if any(hasattr(cb, 'converged') and cb.converged for cb in self.callbacks):
                break
        
        return np.array(trajectory)


class MomentumGradientDescent:
    def __init__(self, energy_model, step_size=0.01, momentum=0.9, 
                 tol=1e-6, callbacks=None):
        self.energy_model = energy_model
        self.step_size = float(step_size)
        self.momentum = float(momentum)
        self.tol = float(tol)
        self.callbacks = callbacks or []
        
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
    
    def minimize(self, x0, max_iter=10000):
        x = np.asarray(x0, dtype=float).copy()
        trajectory = [x.copy()]
        v = np.zeros_like(x)
        
        for step in range(max_iter):
            grad = self.energy_model.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            if not np.isfinite(grad_norm):
                raise RuntimeError(f"Gradient diverged at step {step}")
            
            if grad_norm < self.tol:
                break
        
            v = self.momentum * v - self.step_size * grad
            
            x = x + v
            trajectory.append(x.copy())
            E = self.energy_model.energy(x)
            for cb in self.callbacks:
                cb(step=step, x=x.copy(), energy=E, gradient=grad.copy())
            
            if any(hasattr(cb, 'converged') and cb.converged for cb in self.callbacks):
                break
        
        return np.array(trajectory)