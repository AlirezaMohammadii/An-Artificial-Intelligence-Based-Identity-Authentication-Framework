import os


def count_files_in_subdirectories(directory):
    subdirectories_info = []
    total_files = 0

    # Iterate through subdirectories
    for root, dirs, files in os.walk(directory):
        dir_name = os.path.basename(root)
        files_count = len(files)

        if files_count < 10 or files_count > 10:
            info = f"{dir_name}: {files_count} files"
            subdirectories_info.append(info)

        total_files += files_count

    return subdirectories_info, total_files


if __name__ == "__main__":
    # Example Usage:
    directory_path = "H:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper_my_version/data/sample_dataset/wav"  # Replace with the actual directory path
    try:
        subdirectories_info, total_files = count_files_in_subdirectories(directory_path)

        if subdirectories_info:
            print("\n".join(subdirectories_info))
            print(f"\nTotal number of files in subdirectories: {total_files}")
        else:
            print("No subdirectories found with less or more than 10 files.")
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
