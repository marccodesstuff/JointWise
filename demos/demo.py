import os

# Check if the base directory exists
base_path = "/mnt/f/dataset"
if os.path.exists(base_path):
    print(f"Contents of {base_path}:")
    print(os.listdir(base_path))
else:
    print(f"Directory {base_path} does not exist")

# Check the specific path
target_path = "/mnt/f/dataset/multicoil_train"
print(f"Target path exists: {os.path.exists(target_path)}")