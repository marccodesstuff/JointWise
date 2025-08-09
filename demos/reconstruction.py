import h5py
with h5py.File("/mnt/c/Users/Marc/Downloads/multicoil_train/file1002234.h5", "r") as f:
    print(list(f.keys()))