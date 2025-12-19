import numpy as np




def mean_squared_displacement(trajectory):
    """
    Compute MSD as a function of time.


    trajectory: array of shape (N, d)
    """
    r0 = trajectory[0]
    dr = trajectory - r0
    msd = np.sum(dr**2, axis=1)
    return msd