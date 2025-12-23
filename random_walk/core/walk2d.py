import numpy as np


class RandomWalk2D:
    def __init__(self, n_steps, step_length=1.0, seed=None):
        self.n_steps = int(n_steps)
        self.step_length = float(step_length)
        self.rng = np.random.default_rng(seed)


    def run(self):
        steps = self.rng.choice([-1, 1], size=(self.n_steps, 2))
        positions = np.cumsum(steps, axis=0)
        return self.step_length * positions