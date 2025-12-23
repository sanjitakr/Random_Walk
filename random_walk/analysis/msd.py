import numpy as np

def mean_squared_displacement(trajectory):
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    r0 = trajectory[0]
    dr = trajectory - r0
    msd = np.sum(dr**2, axis=1)
    return msd
