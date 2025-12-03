import os
import random
import shutil
import json
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

def Attacker_generator(directory_path: str, percentage: int) -> None:
    """
    Selects a random subsection of subdirectories within the given directory path,
    renames them to 'attacker_i' where i is an incremental index, and moves them
    to the 'Attackers' subdirectory one level up.

    Args:
    - directory_path (str): The path of the directory containing subdirectories.
    - percentage (int): The percentage of subdirectories to be selected.

    Returns:
    - None
    """
    try:
        # Step 0: Check if 'Attackers' subdirectory already exists
        attackers_directory = os.path.join(directory_path, "..", "Attacker")
        if os.path.exists(attackers_directory) and os.path.isdir(attackers_directory):
            print("The 'Attacker' subdirectory already exists. Exiting the function.")
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

        # Step 5: Rename and move selected subdirectories to 'Attackers'
        for index, subdirectory in enumerate(selected_subdirectories, start=1):
            new_name = f"attacker_{index}"
            old_path = os.path.join(directory_path, subdirectory)
            new_path = os.path.join(attackers_directory, new_name)

            os.rename(old_path, new_path)

        print(
            f"Successfully selected and renamed {num_to_select} subdirectories to 'Attackers'."
        )

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
        mapping_file_path = os.path.join(directory_path, "mapping.json")
        with open(mapping_file_path, "w") as mapping_file:
            json.dump(mapping, mapping_file, indent=2)

    except Exception as e:
        print(f"Error: {e}")


def replace_audio_victim(attacker_directory: str, victim_directory: str) -> None:
    """
    Replace audio files in victim subdirectories with a selection from attacker subdirectories.

    Args:
    - attacker_directory (str): Path to the directory containing attacker subdirectories.
    - victim_directory (str): Path to the directory containing victim subdirectories.

    Returns:
    - None
    """
    try:
        # Step 1: Check if the directories exist
        if (
            not os.path.exists(attacker_directory)
            or not os.path.isdir(attacker_directory)
            or not os.path.exists(victim_directory)
            or not os.path.isdir(victim_directory)
        ):
            raise ValueError("One or both specified directories do not exist.")

        # Step 2: Get the list of attacker and victim subdirectories
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

        # Step 3: Check if there are any attacker or victim subdirectories
        if not attacker_subdirectories:
            print("Error: No attacker subdirectories found.")
            return
        if not victim_subdirectories:
            print("Error: No victim subdirectories found.")
            return

        # Step 4: Iterate through attacker and victim subdirectories
        for attacker_subdir, victim_subdir in zip(
            attacker_subdirectories, victim_subdirectories
        ):
            # Step 5: Get the full paths for the attacker and victim subdirectories
            path_attacker = os.path.join(attacker_directory, attacker_subdir)
            path_victim = os.path.join(victim_directory, victim_subdir)

            # Step 6: Get the list of files in the attacker subdirectory
            files_to_copy = os.listdir(path_attacker)[:5]

            # Step 7: Copy 5 files from the attacker subdirectory to a temporary folder
            temp_folder = os.path.join(attacker_directory, "temp_folder")
            os.makedirs(temp_folder, exist_ok=True)

            for file_name in files_to_copy:
                source_path = os.path.join(path_attacker, file_name)
                destination_path = os.path.join(temp_folder, file_name)
                shutil.copy(source_path, destination_path)

            # Step 8: Delete 5 files from the victim subdirectory
            files_to_delete = os.listdir(path_victim)[:5]
            for file_name in files_to_delete:
                file_path = os.path.join(path_victim, file_name)
                os.remove(file_path)

            # Step 9: Move 5 copied files from the temporary folder to the victim subdirectory
            for file_name in files_to_copy:
                source_path = os.path.join(temp_folder, file_name)
                destination_path = os.path.join(path_victim, file_name)
                shutil.move(source_path, destination_path)

            # Step 10: Remove the temporary folder
            shutil.rmtree(temp_folder)

        # Step 11: Check if there are more attacker subdirectories than victim subdirectories
        if len(attacker_subdirectories) > len(victim_subdirectories):
            remaining_attackers = attacker_subdirectories[len(victim_subdirectories) :]
            print(
                f"\nWarning: There are more attacker subdirectories ({len(remaining_attackers)}) than victim subdirectories. "
                f"Remaining attacker subdirectories: {', '.join(remaining_attackers)}"
            )

    except Exception as e:
        print(f"Error: {e}")


def edit_filenames_in_subdirectories(directory_path):
    # List all subdirectories with names "victim_{i}"
    subdirectories = [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d)) and d.startswith("victim_")
    ]

    for subdirectory in subdirectories:
        subdir_path = os.path.join(directory_path, subdirectory)

        # Get all file names in the subdirectory
        file_names = [
            f
            for f in os.listdir(subdir_path)
            if os.path.isfile(os.path.join(subdir_path, f))
        ]

        if not file_names:
            print(f"No files found in {subdirectory}")
            continue

        # Extract the first part of the name before "-"
        first_part = file_names[0].split("-")[0]

        # Add parentheses to the first part of the name
        new_first_part = f"({first_part})"

        # Rename all files in the subdirectory
        for file_name in file_names:
            original_path = os.path.join(subdir_path, file_name)
            new_name = file_name.replace(file_name.split("-")[0], new_first_part, 1)
            new_path = os.path.join(subdir_path, new_name)

            os.rename(original_path, new_path)
            print(f"Renamed: {file_name} to {new_name}")

def rename_audio_files(base_directory):
    # List all subdirectories in the base directory
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    for i, subdir in enumerate(subdirectories, start=1):
        # Generate a unique 4-digit name for the subdirectory
        new_subdir_name = f"{i:04d}"
        old_subdir_path = os.path.join(base_directory, subdir)
        new_subdir_path = os.path.join(base_directory, new_subdir_name)
        
        # Rename the subdirectory
        os.rename(old_subdir_path, new_subdir_path)
        print(f"Renamed directory: {old_subdir_path} -> {new_subdir_path}")
        
        # Generate a unique 'xxxx' part for all files in this directory
        xxxx = np.random.randint(1000, 9999)
        
        # Process files within the renamed subdirectory
        for j, filename in enumerate(os.listdir(new_subdir_path), start=1):
            if filename.endswith('.wav'):
                # Extract the file extension
                file_ext = os.path.splitext(filename)[1]
                
                # Generate a unique 'yyyy' part for each file
                yyyy = np.random.randint(1000, 9999)
                
                # Create the new filename
                new_filename = f"{new_subdir_name}-{xxxx}-{yyyy}{file_ext}"
                
                # Get the full current path and the full new path
                current_file_path = os.path.join(new_subdir_path, filename)
                new_file_path = os.path.join(new_subdir_path, new_filename)
                
                # Rename the file
                os.rename(current_file_path, new_file_path)
                print(f"Renamed file: {current_file_path} -> {new_file_path}")


def trim_audio_files_in_directory(base_path, start_time, end_time):
    """
    Trims all audio files (.wav, .flac) within the specified directory and its subdirectories
    to the given time range, replacing the original files with the trimmed versions.

    Args:
        base_path (str): Path to the root directory to start searching for audio files.
        start_time (float): Start time in seconds for trimming.
        end_time (float): End time in seconds for trimming.
        
    Raises:
        ValueError: If start_time or end_time is invalid.
    """
    # Validate time range
    if start_time < 0 or end_time <= start_time:
        raise ValueError("Invalid time range specified. Ensure 0 <= start_time < end_time.")
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith((".wav", ".flac")):
                file_path = os.path.join(root, file)
                
                try:
                    # Load the audio file
                    audio = AudioSegment.from_file(file_path)
                    audio_length = len(audio) / 1000  # Convert duration from ms to seconds

                    # Adjust end_time if it exceeds audio length
                    if end_time > audio_length:
                        print(f"Warning: File '{file}' duration ({audio_length}s) "
                              f"is less than the requested end time ({end_time}s). "
                              f"Trimming from {start_time}s to {audio_length}s.")
                        end_time = audio_length

                    # Perform trimming
                    trimmed_audio = audio[start_time * 1000:end_time * 1000]

                    # Overwrite the original file with the trimmed audio
                    trimmed_audio.export(file_path, format=file.split(".")[-1])
                    print(f"Trimmed and replaced: {file_path}")
                
                except CouldntDecodeError:
                    print(f"Error: Could not decode file '{file_path}'. Skipping...")
                except Exception as e:
                    print(f"Error processing file '{file_path}': {e}")

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
