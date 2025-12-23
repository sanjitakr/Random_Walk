import numpy as np
import h5py


def save_hdf5(filename, **arrays):
    with h5py.File(filename, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=np.asarray(arr))
