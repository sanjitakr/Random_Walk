import h5py
import datetime


def save_simulation(filename, trajectory, msd, params):
    with h5py.File(filename, "w") as f:
        f.create_dataset("trajectory", data=trajectory)
        f.create_dataset("msd", data=msd)


        p = f.create_group("parameters")
        for k, v in params.items():
            p.attrs[k] = v


        f.attrs["created"] = datetime.datetime.now().isoformat()