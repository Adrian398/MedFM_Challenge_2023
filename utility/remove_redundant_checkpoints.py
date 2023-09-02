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


files_to_delete = find_pth_files(directory="/scratch/medfm/medfm-challenge/work_dirs")
print("-----------------------------")
print(f"Found {len(files_to_delete)} files.")
print(files_to_delete)

user_input = input("Type 'yes' to confirm deletion: ")
if user_input.lower() == 'yes':
    for file in files_to_delete:
        print("Removing", file)
        try:
            os.remove(file)
        except PermissionError as e:
            print("Failed to remove", file)
else:
    print("Deletion aborted.")