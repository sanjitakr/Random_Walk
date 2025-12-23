import numpy as np
from random_walk import harmonic, GradientDescent


def test_harmonic_minimum():
    energy = harmonic(k=1.0)
    opt = GradientDescent(energy)
    traj = opt.minimize([1.0, -2.0])

    assert np.linalg.norm(traj[-1]) < 1e-3
