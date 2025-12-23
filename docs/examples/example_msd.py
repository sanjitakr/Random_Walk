
import numpy as np
import matplotlib.pyplot as plt
import h5py
from random_walk.core.walk1d import RandomWalk1D
from random_walk.analysis.msd import mean_squared_displacement

n_steps = 100
n_trajectories = 500    
step_length = 0.5
seed = 42
png_file = "msd_plot.png"
hdf5_file = "msd_data.h5"

msds = []
trajectories = []

rng = np.random.default_rng(seed)

for i in range(n_trajectories):
    rw = RandomWalk1D(n_steps=n_steps, step_length=step_length, seed=rng.integers(1e6))
    pos = rw.run()
    trajectories.append(pos)
    msds.append(mean_squared_displacement(pos))

msds = np.array(msds)
avg_msd = np.mean(msds, axis=0)


plt.figure(figsize=(8, 4))
plt.plot(avg_msd, label=f"Average MSD over {n_trajectories} trajectories")
plt.xlabel("Step")
plt.ylabel("MSD")
plt.title("1D Random Walk Mean Squared Displacement")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(png_file)
plt.show()


with h5py.File(hdf5_file, "w") as f:
    f.create_dataset("trajectories", data=np.array(trajectories))
    f.create_dataset("msd", data=avg_msd)

print(f"Saved MSD plot as '{png_file}' and data as '{hdf5_file}'")
