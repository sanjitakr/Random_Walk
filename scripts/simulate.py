from randomwalk import RandomWalk2D
from randomwalk.analysis.msd import mean_squared_displacement
from randomwalk.io.hdf5 import save_simulation


rw = RandomWalk2D(
    n_steps=100_000,
    step_length=1.0,
    seed=42
    )


trajectory = rw.run()
msd = mean_squared_displacement(trajectory)


params = {
    "dimension": 2,
    "n_steps": 100_000,
    "step_length": 1.0,
    "seed": 42,
}


save_simulation(
    "rw_2d_steps100k_seed42.h5",
    trajectory,
    msd,
    params
)