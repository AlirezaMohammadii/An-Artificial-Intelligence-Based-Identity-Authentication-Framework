# git push -f origin main
import os
import shutil
import random
import concurrent.futures
from pydub import AudioSegment
from pathlib import Path


def get_all_files(directory):
    """
    Get a list of all file paths in the specified directory and its subdirectories.

    Args:
        directory (str): The path to the directory.

    Returns:
        list: A list of file paths.
    """
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def has_exact_file_count(directory, count=10):
    """
    Check if each subdirectory within the parent directory has exactly 'count' audio files.

    Args:
        directory (str): The path to the parent directory.
        count (int): The desired number of audio files.

    Returns:
        bool: True if all subdirectories have exact 'count' audio files, False otherwise.
    """
    for subdirectory in os.listdir(directory):
        subdirectory_path = os.path.join(directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            audio_files = [
                f
                for f in os.listdir(subdirectory_path)
                if f.lower().endswith((".flac", ".wav"))
            ]
            if len(audio_files) > count:
                return False
    return True


def prepare_destination_directories(parent_directory):
    """
    Create subdirectories in the parent directory based on existing directories.

    Args:
        parent_directory (str): The path to the parent directory.
    """

    if has_exact_file_count(parent_directory, count=10):
        print(
            "All subdirectories already contain exactly 10 audio files. Skipping the process."
        )
        return

    subdirectories = [
        d
        for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d))
    ]
    for subdirectory in subdirectories:
        destination_path = os.path.join(parent_directory, subdirectory)
        os.makedirs(destination_path, exist_ok=True)


def move_files_to_destinations(files, parent_directory):
    """
    Move files to destination directories based on their immediate parent directory.

    Args:
        files (list): List of file paths.
        parent_directory (str): The path to the parent directory.
    """
    for file in files:
        destination_directory = os.path.join(
            parent_directory, file.split(os.path.sep)[1]
        )
        destination_path = os.path.join(destination_directory, os.path.basename(file))
        os.replace(file, destination_path)


def delete_short_audio_files(directory, duration_threshold=10):
    """
    Delete audio files shorter than a specified duration.

    Args:
        directory (str): The path to the directory.
        duration_threshold (int): The minimum duration threshold in seconds.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".flac", ".wav")):
                    file_path = os.path.join(root, file)
                    futures.append(
                        executor.submit(
                            process_audio_file, file_path, duration_threshold
                        )
                    )

        # Wait for all threads to complete
        concurrent.futures.wait(futures)


def process_audio_file(file_path, duration_threshold):
    """
    Process audio files and delete those below the duration threshold.

    Args:
        file_path (Path): The path to the audio file.
        duration_threshold (int): The minimum duration threshold in seconds.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        duration_in_seconds = len(audio) / 1000

        if duration_in_seconds < duration_threshold:
            os.remove(file_path)
    except FileNotFoundError:
        pass  # Handle file not found exception
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def delete_matrices(directory_path):
    """
    Delete directories and files with the name 'matrices'.

    Args:
        directory_path (str): The path to the directory.
    """
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for item in dirs + files:
            item_path = os.path.join(root, item)
            if item.lower() == "matrices":
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted directory: {item_path}")
                else:
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")


def delete_files_except_flac(directory_path):
    """
    Delete files in a directory except those with the '.flac' extension.

    Args:
        directory_path (str): The path to the directory.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.endswith(".flac"):
                os.remove(file_path)


def keep_random_files(directory_path, seed=None):
    """
    Randomly keeps exactly 10 '.flac' files in each subdirectory within the given main directory,
    and deletes the rest. The randomization is reproducible if a seed is provided.

    Parameters:
        main_directory (str): The main directory path.
        seed (int): Seed for reproducibility in randomization. Default is None.

    Raises:
        ValueError: If the provided main_directory is not a valid directory.

    Returns:
        None
    """
    # Set seed for reproducibility
    random.seed(43)

    # Validate and convert input to Path object
    main_directory = Path(directory_path)
    if not main_directory.is_dir():
        raise ValueError("The provided main_directory is not a valid directory.")

    # Iterate through subdirectories
    for subdirectory in main_directory.iterdir():
        if subdirectory.is_dir():
            # Get all .flac files in the subdirectory
            flac_files = list(subdirectory.glob("*.flac"))

            # Check if there are more than 10 .flac files
            if len(flac_files) > 10:
                # Randomly select 10 files
                files_to_keep = random.sample(flac_files, 10)

                # Delete the rest of the files
                for file in flac_files:
                    if file not in files_to_keep:
                        file.unlink()


def delete_empty_subdirectories(directory_path):
    """
    Delete empty subdirectories in the specified directory.

    Args:
        directory_path (str): The path to the directory.
    """
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


def count_files_in_subdirectories(directory):
    total_files = 0

    # Iterate through subdirectories
    for root, dirs, files in os.walk(directory):
        total_files += len(files)

    return total_files


def main(parent_directory):
    print("Calculating the quantity of subdirectories and their audio files...")
    subdirectory_count = 0
    subdirectories_with_more_than_10 = 0

    for subdirectory in os.listdir(parent_directory):
        subdirectory_path = os.path.join(parent_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            audio_files = [
                f
                for f in os.listdir(subdirectory_path)
                if f.lower().endswith((".flac", ".wav"))
            ]
            audio_file_count = len(audio_files)

            if audio_file_count > 10:
                subdirectories_with_more_than_10 += 1

            if audio_file_count > 0:
                subdirectory_count += 1

    print(f"Total subdirectories found: {subdirectory_count}")
    print(
        f"Subdirectories with more than 10 audio files: {subdirectories_with_more_than_10}"
    )

    # Get all files, prepare directories, and move files to destinations
    all_files = get_all_files(parent_directory)
    prepare_destination_directories(parent_directory)
    move_files_to_destinations(all_files, parent_directory)

    if subdirectories_with_more_than_10 > 100:
        print("More than 40 subdirectories found with more than 10 audio files.")
        print("Performing deletion and randomization in batches of 10...")

        # Calculate how many batches of 10 subdirectories are needed
        batches = (subdirectories_with_more_than_10 + 9) // 10

        for batch in range(batches):
            start_idx = batch * 10
            end_idx = min((batch + 1) * 10, subdirectories_with_more_than_10)

            print(f"Processing batch {batch + 1} ({start_idx + 1} to {end_idx})...")

            # Perform operations on this batch of subdirectories
            delete_short_audio_files(parent_directory, duration_threshold=10)
            keep_random_files(parent_directory, seed=43)
            delete_files_except_flac(parent_directory)
            delete_empty_subdirectories(parent_directory)

            print(f"Batch {batch + 1} completed.")

    elif subdirectories_with_more_than_10 == 0:
        print("No subdirectory has more than 10 audio files. Skipping the process.")
    else:
        # If there are 40 or fewer subdirectories with more than 10 audio files,
        # perform operations on all of them at once
        print(
            "40 or fewer subdirectories found with more than 10 audio files."
            " Performing operations on all subdirectories..."
        )
        delete_short_audio_files(parent_directory, duration_threshold=10)
        keep_random_files(parent_directory, seed=43)
        delete_files_except_flac(parent_directory)
        delete_empty_subdirectories(parent_directory)

    print("\nTrimming voices has finished.")


if __name__ == "__main__":
    # Example Usage:
    parent_dir = "H:/1.Deakin university/Python/test"
    import time

    start_time = time.time()
    main(parent_dir)
    end_time = time.time()

    # Print execution time
    print(f"Execution time: {end_time - start_time} seconds")