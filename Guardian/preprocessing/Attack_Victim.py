import os
import random
import shutil
import json
from itertools import cycle
from random import sample


def Attacker_generator(directory_path: str, percentage: int) -> None:
    """
    Selects a random subsection of subdirectories within the given directory path,
    renames them to 'attacker_i' where i is an incremental index, and moves them
    to the 'Attackers' subdirectory one level up. Additionally, it outputs a mapping
    between the original names of the folders and their new names.

    Args:
    - directory_path (str): The path of the directory containing subdirectories.
    - percentage (int): The percentage of subdirectories to be selected.

    Returns:
    - None
    """
    try:
        # Step 0: Check if 'Attackers' subdirectory already exists
        attackers_directory = os.path.join(directory_path, "..", "Attackers")
        if os.path.exists(attackers_directory) and os.path.isdir(attackers_directory):
            print("The 'Attackers' subdirectory already exists. Exiting the function.")
            return

        # Step 1: Validate and get the list of subdirectories
        if not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path.")

        subdirectories = [
            d
            for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d))
        ]

        # Step 2: Create 'Attackers' subdirectory if it doesn't exist
        os.makedirs(attackers_directory, exist_ok=True)

        # Step 3: Calculate the number of subdirectories to select
        num_to_select = int(len(subdirectories) * (percentage / 100.0))

        # Step 4: Randomly select subdirectories
        selected_subdirectories = random.sample(subdirectories, num_to_select)

        # Step 5: Initialize mapping dictionary
        mapping = {}

        # Step 6: Rename and move selected subdirectories to 'Attackers'
        for index, subdirectory in enumerate(selected_subdirectories, start=1):
            new_name = f"attacker_{index}"
            old_path = os.path.join(directory_path, subdirectory)
            new_path = os.path.join(attackers_directory, new_name)

            # Update mapping
            mapping[subdirectory] = new_name

            os.rename(old_path, new_path)

        print(
            f"Successfully selected and renamed {num_to_select} subdirectories to 'Attackers'."
        )

        # Step 7: Save the mapping as a dictionary in a JSON format file
        mapping_file_path = os.path.join(attackers_directory, "attackers.json")
        with open(mapping_file_path, "w") as mapping_file:
            json.dump(mapping, mapping_file, indent=2)

    except Exception as e:
        print(f"Error: {e}")


def new_user_generator(directory_path: str, percentage: int) -> None:
    """
    Selects a random subsection of subdirectories within the given directory path,
    renames them to 'new_user_i' where i is an incremental index, and moves them
    to the 'new_user' subdirectory one level up. Additionally, it outputs a mapping
    between the original names of the folders and their new names.

    Args:
    - directory_path (str): The path of the directory containing subdirectories.
    - percentage (int): The percentage of subdirectories to be selected.

    Returns:
    - None
    """
    try:
        # Step 0: Check if 'new_user' subdirectory already exists
        new_user_directory = os.path.join(directory_path, "..", "new_user")
        if os.path.exists(new_user_directory) and os.path.isdir(new_user_directory):
            print("The 'new_user' subdirectory already exists. Exiting the function.")
            return

        # Step 1: Validate and get the list of subdirectories
        if not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path.")

        subdirectories = [
            d
            for d in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, d))
        ]

        # Step 2: Create 'new_user' subdirectory if it doesn't exist
        os.makedirs(new_user_directory, exist_ok=True)

        # Step 3: Calculate the number of subdirectories to select
        num_to_select = int(len(subdirectories) * (percentage / 100.0))

        # Step 4: Randomly select subdirectories
        selected_subdirectories = random.sample(subdirectories, num_to_select)

        # Step 5: Initialize mapping dictionary
        mapping = {}

        # Step 6: Rename and move selected subdirectories to 'new_user'
        for index, subdirectory in enumerate(selected_subdirectories, start=1):
            new_name = f"new_user_{index}"
            old_path = os.path.join(directory_path, subdirectory)
            new_path = os.path.join(new_user_directory, new_name)

            # Update mapping
            mapping[subdirectory] = new_name

            os.rename(old_path, new_path)

        print(
            f"\n Successfully selected and renamed {num_to_select} subdirectories to 'new_user'."
        )

        # Step 7: Save the mapping as a dictionary in a JSON format file
        mapping_file_path = os.path.join(new_user_directory, "new_user.json")
        with open(mapping_file_path, "w") as mapping_file:
            json.dump(mapping, mapping_file, indent=2)

    except Exception as e:
        print(f"Error: {e}")


def rename_victim_subdirectories(directory_path: str, selected_percentage: int) -> None:
    """
    Rename a randomly selected subsection of subdirectories within the given directory path.
    Create a mapping between the original names and the new names in "victim_{i}" format.
    Save the mapping as a dictionary in a JSON format file.

    Args:
    - directory_path (str): The path to the target directory.
    - selected_percentage (int): The percentage of subdirectories to be renamed.

    Returns:
    - None
    """
    try:
        # Step 1: Check if the directory path exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path provided.")

        # Step 2: Get the list of subdirectories
        subdirectories = [
            subdir
            for subdir in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, subdir))
        ]

        # Step 3: Calculate the number of subdirectories to be renamed based on the selected percentage
        num_subdirectories_to_rename = int(
            len(subdirectories) * (selected_percentage / 100)
        )

        # Step 4: Randomly select the subsection of subdirectories
        selected_subdirectories = random.sample(
            subdirectories, num_subdirectories_to_rename
        )

        # Step 5: Rename the selected subdirectories
        mapping = {}
        for i, subdirectory in enumerate(selected_subdirectories, start=1):
            old_path = os.path.join(directory_path, subdirectory)
            new_name = f"victim_{i}"
            new_path = os.path.join(directory_path, new_name)

            mapping[subdirectory] = new_name
            os.rename(old_path, new_path)

        # Step 6: Print a summary message
        print(
            f"Successfully renamed {num_subdirectories_to_rename} subdirectories in '{directory_path}'."
        )

        # Step 7: Save the mapping as a dictionary in a JSON format file
        mapping_file_path = os.path.join(directory_path, "victims.json")
        with open(mapping_file_path, "w") as mapping_file:
            json.dump(mapping, mapping_file, indent=2)

    except Exception as e:
        print(f"Error: {e}")


def replace_audio_victim(
    attacker_directory: str,
    victim_directory: str,
    number_of_replacements: int = 5,
    mode: str = "sequential",
) -> None:
    """
    Replace audio files in victim subdirectories with a selection from attacker subdirectories based on a specified number
    of replacements and a mode (sequential or random).
    Outputs a mapping in a format which shows which "attacker_{i}" replaced the "victim_{x}".

    Args:
    - attacker_directory (str): Path to the directory containing attacker subdirectories.
    - victim_directory (str): Path to the directory containing victim subdirectories.
    - number_of_replacements (int): Number of files to replace, must be in the range 1 to 10.
    - mode (str): Replacement mode, "sequential" or "random" for the selection of files.

    Returns:
    - None
    """

    try:
        if (
            not os.path.exists(attacker_directory)
            or not os.path.isdir(attacker_directory)
            or not os.path.exists(victim_directory)
            or not os.path.isdir(victim_directory)
        ):
            raise ValueError("One or both specified directories do not exist.")

        if not 1 <= number_of_replacements < 10:
            raise ValueError("Number of replacements must be in the range 1 to 10.")

        attacker_subdirectories = [
            subdir
            for subdir in os.listdir(attacker_directory)
            if os.path.isdir(os.path.join(attacker_directory, subdir))
            and subdir.startswith("attacker_")
        ]
        victim_subdirectories = [
            subdir
            for subdir in os.listdir(victim_directory)
            if os.path.isdir(os.path.join(victim_directory, subdir))
            and subdir.startswith("victim_")
        ]

        if not attacker_subdirectories or not victim_subdirectories:
            raise ValueError("Error: No attacker or victim subdirectories found.")

        attacker_cycle = cycle(attacker_subdirectories)
        replacement_mapping = {}  # Initialize mapping dictionary

        for victim_subdir in victim_subdirectories:
            attacker_subdir = next(attacker_cycle)
            replacement_mapping[victim_subdir] = attacker_subdir  # Update mapping

            # Paths for attacker and victim
            path_attacker = os.path.join(attacker_directory, attacker_subdir)
            path_victim = os.path.join(victim_directory, victim_subdir)

            if mode == "random":
                files_in_attacker_dir = os.listdir(path_attacker)
                files_to_copy = sample(files_in_attacker_dir, k=number_of_replacements)

                files_in_victim_dir = os.listdir(path_victim)
                files_to_delete = sample(files_in_victim_dir, k=number_of_replacements)
            else:  # mode == "sequential"
                files_to_copy = os.listdir(path_attacker)[:number_of_replacements]
                files_to_delete = os.listdir(path_victim)[:number_of_replacements]

            temp_folder = os.path.join(attacker_directory, "temp_folder")
            os.makedirs(temp_folder, exist_ok=True)

            for file_name in files_to_copy:
                source_path = os.path.join(path_attacker, file_name)
                destination_path = os.path.join(temp_folder, file_name)
                shutil.copy(source_path, destination_path)

            for file_name in files_to_delete:
                file_path = os.path.join(path_victim, file_name)
                os.remove(file_path)

            for file_name in files_to_copy:
                source_path = os.path.join(temp_folder, file_name)
                destination_path = os.path.join(path_victim, file_name)
                shutil.move(source_path, destination_path)

            shutil.rmtree(temp_folder)

        # After processing all replacements, save the mapping to a JSON file
        mapping_file_path = os.path.join(victim_directory, "replacements.json")
        with open(mapping_file_path, "w") as file:
            json.dump(replacement_mapping, file, indent=4)

        print(f"Replacement mapping saved to {mapping_file_path}")
        print(
            f"\n Utterances got replaced in each victim's account: {number_of_replacements} - mode: {mode}"
        )

    except Exception as e:
        print(f"Error: {e}")


# Commented out function call for safety
# replace_audio_victim("/path/to/attacker_directory", "/path/to/victim_directory", 5, "random")


def edit_filenames_in_subdirectories(directory_path):
    # Path to the victims.json file
    victims_json_path = os.path.join(directory_path, "victims.json")

    # Load the mapping from the JSON file
    with open(victims_json_path, "r") as file:
        victim_mapping = json.load(file)

    # Reverse the mapping to get folder names as keys and numbers as values
    folder_to_key_mapping = {v: k for k, v in victim_mapping.items()}

    # List all subdirectories with names "victim_{i}"
    subdirectories = [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d)) and d.startswith("victim_")
    ]

    for subdirectory in subdirectories:
        subdir_path = os.path.join(directory_path, subdirectory)

        # Check if the subdirectory is in the mapping
        if subdirectory in folder_to_key_mapping:
            # Get the corresponding key for the folder
            key = folder_to_key_mapping[subdirectory]

            # Get all file names in the subdirectory
            file_names = [
                f
                for f in os.listdir(subdir_path)
                if os.path.isfile(os.path.join(subdir_path, f))
            ]

            if not file_names:
                print(f"No files found in {subdirectory}")
                continue

            # Rename all files in the subdirectory using the key
            for file_name in file_names:
                original_path = os.path.join(subdir_path, file_name)
                new_name = file_name.replace(file_name.split("-")[0], f"({key})", 1)
                new_path = os.path.join(subdir_path, new_name)

                os.rename(original_path, new_path)
                print(f"Renamed: {file_name} to {new_name}")


if __name__ == "__main__":
    # Example Usage:
    directory_path = (
        "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Attacker-Victim"
    )
    percentage_attacker = 5
    percentage_victim = 5
    Attacker_generator(directory_path, percentage_attacker)
    rename_victim_subdirectories(directory_path, percentage_victim)

    # Get the parent directory paths for attackers and victims from the user
    attacker_directory = (
        "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Attackers"
    )
    victim_directory = (
        "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Attacker-Victim/"
    )
    replace_audio_victim(attacker_directory, victim_directory)
    edit_filenames_in_subdirectories(directory_path)
