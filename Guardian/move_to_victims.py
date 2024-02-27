import os
import random
import shutil
import json


def move_subdirectories_to_victims_directory(directory_path: str) -> None:
    """
    Create a 'Victims' directory and move the selected subdirectories into it.
    Move the mapping file into the 'Victims' directory.

    Args:
    - directory_path (str): The path to the target directory.

    Returns:
    - None
    """
    try:
        # Step 1: Check if the directory path exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path provided.")

        # Step 2: Create 'Victims' directory if it doesn't exist
        victims_directory = os.path.join(directory_path, "..", "Victims")
        os.makedirs(victims_directory, exist_ok=True)

        # Step 3: Get the list of subdirectories in the provided directory
        subdirectories = [
            subdir
            for subdir in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, subdir))
            and subdir.startswith("victim_")
        ]

        # Step 4: Move the subdirectories into 'Victims' directory
        for subdirectory in subdirectories:
            old_path = os.path.join(directory_path, subdirectory)
            new_path = os.path.join(victims_directory, subdirectory)
            shutil.move(old_path, new_path)

        # Step 5: Move the mapping file into 'Victims' directory
        mapping_file_path = os.path.join(directory_path, "mapping.json")
        new_mapping_file_path = os.path.join(victims_directory, "mapping.json")
        shutil.move(mapping_file_path, new_mapping_file_path)

        print(
            f"Successfully moved subdirectories and mapping file to '{victims_directory}'."
        )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    directory_path = (
        "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Attacker-Victim"
    )
    # Move subdirectories and mapping file to 'Victims' directory
    move_subdirectories_to_victims_directory(directory_path)
