import numpy as np

class Callback:
    def __call__(self, step, x, energy, gradient):
        raise NotImplementedError

class PrintEnergy(Callback):
    def __init__(self, every=100):
        self.every = int(every)
    
    def __call__(self, step, x, energy, gradient):
        if step % self.every == 0:
            grad_norm = np.linalg.norm(gradient)
            print(f"Step {step:6d} | Energy = {energy:.6e} | |∇E| = {grad_norm:.6e}")

class TrajectoryRecorder(Callback):
    def __init__(self):
        self.trajectory = []
        self.energies = []
        self.gradients = []
    
    def __call__(self, step, x, energy, gradient):
        self.trajectory.append(x.copy())
        self.energies.append(energy)
        self.gradients.append(gradient.copy())
    
    def get_trajectory(self):
        return np.array(self.trajectory)
    
    def get_energies(self):
        return np.array(self.energies)
    
    def get_gradients(self):
        return np.array(self.gradients)
    
    def clear(self):
        self.trajectory = []
        self.energies = []
        self.gradients = []

class EnergyConvergence(Callback):
    def __init__(self, tol=1e-8, patience=10):
        self.tol = float(tol)
        self.patience = int(patience)
        self.prev_energy = None
        self.converged = False
        self.below_tol_count = 0
    
    def __call__(self, step, x, energy, gradient):
        if self.prev_energy is not None:
            energy_change = abs(energy - self.prev_energy)
            
            if energy_change < self.tol:
                self.below_tol_count += 1
                if self.below_tol_count >= self.patience:
                    self.converged = True
            else:
                self.below_tol_count = 0
        
        self.prev_energy = energy
    
    def reset(self):
        self.prev_energy = None
        self.converged = False
        self.below_tol_count = 0


class GradientNorm(Callback):
    def __init__(self):
        self.norms = []
        self.steps = []
    
    def __call__(self, step, x, energy, gradient):
        norm = np.linalg.norm(gradient)
        self.norms.append(norm)
        self.steps.append(step)
    
    def get_norms(self):
        return np.array(self.norms)
    
    def get_steps(self):
        return np.array(self.steps)
    
    def clear(self):
        self.norms = []
        self.steps = []


class LineSearchMonitor(Callback):
    def __init__(self):
        self.step_sizes = []
    
    def record_step_size(self, step_size):
        self.step_sizes.append(step_size)
    
    def __call__(self, step, x, energy, gradient):
        pass
    
    def get_step_sizes(self):
        return np.array(self.step_sizes)