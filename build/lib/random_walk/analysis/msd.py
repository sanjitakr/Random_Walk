import numpy as np




def mean_squared_displacement(trajectory):
    r0 = trajectory[0]
    dr = trajectory - r0
    msd = np.sum(dr**2, axis=1)
    return msd