import os
import csv

def check_filenames_in_csv(directory, csv_file, csv_column=0):
    # Read filenames from CSV
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        csv_filenames = set(row[csv_column] for row in reader if row)
    
    # List files in directory
    for filename in os.listdir(directory):
        name, _ = os.path.splitext(filename)
        if name in csv_filenames:
            print(f"{filename} is present in CSV")
        else:
            print(f"{filename} is NOT present in CSV")

# Example usage:
check_filenames_in_csv('/mnt/c/Users/Marc/Downloads/multicoil_train', '../annotations/knee-list.csv')