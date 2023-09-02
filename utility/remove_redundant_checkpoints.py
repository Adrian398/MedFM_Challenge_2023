import os


def find_pth_files(directory="."):
    """Find all .pth files in the directory and its subdirectories that don't start with 'best'."""
    matching_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            print(f"Checking {file}")
            if file.endswith('.pth') and not file.startswith('best'):
                matching_files.append(os.path.join(root, file))

    return matching_files


files = find_pth_files(directory="/scratch/medfm/medfm-challenge/work_dirs")
print("-----------------------------")
print(f"Found {len(files)} files.")
print(files)
