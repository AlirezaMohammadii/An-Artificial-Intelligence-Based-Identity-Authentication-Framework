import os
from pydub import AudioSegment
import shutil
import random
import concurrent.futures
from pydub import AudioSegment


def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def prepare_destination_directories(parent_directory):
    subdirectories = [
        d
        for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d))
    ]
    for subdirectory in subdirectories:
        destination_path = os.path.join(parent_directory, subdirectory)
        os.makedirs(destination_path, exist_ok=True)


def move_files_to_destinations(files, parent_directory):
    for file in files:
        destination_directory = os.path.join(
            parent_directory, file.split(os.path.sep)[1]
        )
        destination_path = os.path.join(destination_directory, os.path.basename(file))
        os.replace(file, destination_path)


def delete_short_audio_files(directory, duration_threshold=7):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for root, dirs, files in os.walk(directory):
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
    try:
        audio = AudioSegment.from_file(file_path)
        duration_in_seconds = len(audio) / 1000

        if duration_in_seconds < duration_threshold:
            os.remove(file_path)
    except Exception as e:
        # Handle exceptions (e.g., invalid audio file)
        pass


def delete_matrices(directory_path):
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for directory in dirs:
            if directory.lower() == "matrices":
                matrices_path = os.path.join(root, directory)
                shutil.rmtree(matrices_path)
                print(f"Deleted directory: {matrices_path}")

        for file in files:
            if file.lower() == "matrices":
                matrices_path = os.path.join(root, file)
                os.remove(matrices_path)
                print(f"Deleted file: {matrices_path}")


def delete_files_except_flac(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.endswith(".flac"):
                print(f"Deleting: {file_path}")
                os.remove(file_path)


def keep_random_files(directory, num_files_to_keep=10):
    for root, dirs, files in os.walk(directory):
        if files and any(file.lower().endswith(".flac") for file in files):
            files_to_keep = random.sample(files, min(num_files_to_keep, len(files)))

            for file in files:
                file_path = os.path.join(root, file)
                if file not in files_to_keep:
                    print(f"Deleting {file_path}")
                    os.remove(file_path)


def delete_empty_subdirectories(directory_path):
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):
                print(f"Deleting empty directory: {dir_path}")
                os.rmdir(dir_path)


def main(parent_directory):
    all_files = get_all_files(parent_directory)
    prepare_destination_directories(parent_directory)
    move_files_to_destinations(all_files, parent_directory)

    # Additional functions
    # delete_short_audio_files(parent_directory, duration_threshold=7)
    keep_random_files(parent_directory)
    delete_matrices(parent_directory)
    delete_files_except_flac(parent_directory)
    delete_empty_subdirectories(parent_directory)


# Example Usage:
parent_dir = (
    "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Deep-VR/dev-clean-original"
)
main(parent_dir)
