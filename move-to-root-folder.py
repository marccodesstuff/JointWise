import os
import shutil
import sys

def move_subfolders_to_root(input_dir):
    input_dir = os.path.abspath(input_dir)
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                target_path = os.path.join(input_dir, sub_item)
                if os.path.exists(target_path):
                    print(f"Skipping '{sub_item_path}' (target exists)")
                    continue
                shutil.move(sub_item_path, target_path)
            # Remove the folder even if not empty
            shutil.rmtree(item_path)

if __name__ == "__main__":
    move_subfolders_to_root("/home/bictor0301/Code/JointWise/output-folder")
