import numpy as np
from random_walk import RandomWalk1D


def test_random_walk_length():
    rw = RandomWalk1D(100)
    traj = rw.run()
    assert len(traj) == 100


def test_random_walk_step_size():
    rw = RandomWalk1D(100, step_length=2.0)
    traj = rw.run()
    steps = np.diff(np.concatenate([[0], traj]))
    assert np.all(np.abs(steps) == 2.0)
