import os
import shutil
import json


def move_files_to_poisoned_data_pool(directory_path: str) -> None:
    try:
        victims_directory = os.path.join(directory_path, "Victims")
        poisoned_data_pool_directory = os.path.join(
            directory_path, "Poisoned data pool"
        )

        # Check if 'Victims' directory exists
        if not os.path.exists(victims_directory) or not os.path.isdir(
            victims_directory
        ):
            print("Error: 'Victims' directory does not exist.")
            return

        # Check if 'Poisoned data pool' directory exists, create if not
        os.makedirs(poisoned_data_pool_directory, exist_ok=True)

        # Find the largest index for "victim_{i}" subdirectories in 'Poisoned data pool'
        existing_victim_indices = set()
        for item in os.listdir(poisoned_data_pool_directory):
            if os.path.isdir(
                os.path.join(poisoned_data_pool_directory, item)
            ) and item.startswith("victim_"):
                try:
                    index = int(item.split("_")[1])
                    existing_victim_indices.add(index)
                except ValueError:
                    pass

        if existing_victim_indices:
            largest_index = max(existing_victim_indices)
        else:
            largest_index = 0

        # Move files and subdirectories from 'Victims' to 'Poisoned data pool'
        for item in os.listdir(victims_directory):
            source_path = os.path.join(victims_directory, item)

            # Check if the item is a file or subdirectory
            if os.path.isfile(source_path):
                # Rename files with "victim_{k}" format
                largest_j = get_largest_mapping_index(poisoned_data_pool_directory)
                new_name = f"victim_{largest_index + 1}_{item}"
            else:
                # Rename subdirectories with "victim_{k}" format
                new_name = f"victim_{largest_index + 1}_{item}"

            destination_path = os.path.join(poisoned_data_pool_directory, new_name)

            shutil.move(source_path, destination_path)

        print("Successfully moved files and subdirectories to 'Poisoned data pool'.")
        shutil.rmtree(victims_directory)
    except Exception as e:
        print(f"Error: {e}")


def get_largest_mapping_index(poisoned_data_pool_directory: str) -> int:
    # Find the largest index for "mapping_{j}.json" files in 'Poisoned data pool'
    existing_mapping_indices = set()
    for item in os.listdir(poisoned_data_pool_directory):
        if (
            os.path.isfile(os.path.join(poisoned_data_pool_directory, item))
            and item.startswith("mapping_")
            and item.endswith(".json")
        ):
            try:
                index = int(item.split("_")[1].split(".")[0])
                existing_mapping_indices.add(index)
            except ValueError:
                pass

    if existing_mapping_indices:
        return max(existing_mapping_indices)
    else:
        return 0


if __name__ == "__main__":
    directory_path = "F:/1.Deakin university/Python/13_10_2023_My_Project_1/"
    move_files_to_poisoned_data_pool(directory_path)
