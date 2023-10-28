import os
import shutil

WORK_DIR_SOURCE = "/scratch/medfm/medfm-challenge/work_dirs"
WORK_DIR_TARGET = "~/Git/medfm-challenge/work_dirs"


def copy_specific_files(src_root, dst_root):
    # Define the subdirectories structure
    first_level_subdirs = ["colon", "endo", "chest"]
    second_level_subdirs = ["1-shot", "5-shot", "10-shot"]

    for first_level in first_level_subdirs:
        for second_level in second_level_subdirs:
            # Source and destination directories
            src_dir = os.path.join(src_root, first_level, second_level)
            dst_dir = os.path.join(dst_root, first_level, second_level)

            # Ensure the source directory exists
            if not os.path.exists(src_dir):
                print(f"Directory {src_dir} doesn't exist. Skipping...")
                continue

            # Create the destination directory if it doesn't exist
            os.makedirs(dst_dir, exist_ok=True)
            print(dst_dir)

            # Copy specific files
            for filename in os.listdir(src_dir):
                if (filename.endswith(".py") or
                        filename.endswith("submission.csv") or
                        filename.endswith("validation.csv") or
                        filename == "performance.json"):

                    src_file = os.path.join(src_dir, filename)
                    dst_file = os.path.join(dst_dir, filename)

                    try:
                        shutil.copy2(src_file, dst_file)
                        print(f"Copied: {src_file} to {dst_file}")
                    except Exception as e:
                        print(f"Error copying {src_file} to {dst_file}. Error: {e}")


if __name__ == "__main__":
    #src_root = input("Enter the path to the root directory: ").strip()
    #dst_root = input("Enter the destination directory: ").strip()

    src_root = WORK_DIR_SOURCE
    dst_root = WORK_DIR_TARGET

    copy_specific_files(src_root, dst_root)
    print("Done!")
