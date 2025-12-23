import h5py


def load_hdf5(filename):
    data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            data[key] = f[key][:]
    return data
