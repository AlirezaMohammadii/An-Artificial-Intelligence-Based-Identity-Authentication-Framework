import os
import shutil
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm


def process_audio_file(file_path, duration_threshold):
    """
    Process an audio file to check if its duration is above a certain threshold.
    Deletes the file if it does not meet the threshold.

    Args:
        file_path (str): The path to the audio file.
        duration_threshold (int): The duration threshold in seconds.
    """
    try:
        audio = AudioSegment.from_file(file_path)
        duration_in_seconds = len(audio) / 1000
        if duration_in_seconds < duration_threshold:
            os.remove(file_path)
            return file_path, False
        else:
            return file_path, True
    except Exception as e:
        return file_path, e


def delete_short_audio_files(directory, duration_threshold=10):
    """
    Delete audio files shorter than a specified duration in parallel.

    Args:
        directory (str): The path to the directory.
        duration_threshold (int): The minimum duration threshold in seconds.
    """
    audio_files = [
        str(file)
        for file in Path(directory).rglob("*")
        if file.suffix.lower() in [".flac", ".wav"]
    ]
    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_audio_file, file, duration_threshold): file
            for file in audio_files
        }
        for future in tqdm(
            as_completed(future_to_file),
            total=len(audio_files),
            desc="Processing Audio Files",
        ):
            file, result = future.result()


def move_audio_files_to_parent_directory(directory):
    """
    Moves audio files from child directories to their corresponding parent subdirectory.
    """
    for subdirectory in tqdm(Path(directory).iterdir(), desc="Organizing Files"):
        if subdirectory.is_dir():
            for child in subdirectory.iterdir():
                if child.is_dir():
                    for audio_file in child.glob("*.*"):
                        if audio_file.suffix.lower() in [".flac", ".wav"]:
                            shutil.move(
                                str(audio_file), str(subdirectory / audio_file.name)
                            )
                    shutil.rmtree(str(child))


def delete_non_audio_files(directory):
    """
    Deletes files except audio files within each subdirectory of the provided directory.
    """
    for file in tqdm(Path(directory).rglob("*.*"), desc="Cleaning Up Non-Audio Files"):
        if file.is_file() and file.suffix.lower() not in [".flac", ".wav"]:
            file.unlink()


def keep_random_files(directory_path, seed=None):
    """
    Randomly keeps exactly 10 '.flac' files in each subdirectory within the given main directory,
    and deletes the rest.
    """
    random.seed(seed)
    for subdirectory in tqdm(
        Path(directory_path).iterdir(), desc="Selecting Random Files"
    ):
        if subdirectory.is_dir():
            flac_files = list(subdirectory.glob("*.flac"))
            if len(flac_files) > 10:
                files_to_keep = random.sample(flac_files, 10)
                for file in flac_files:
                    if file not in files_to_keep:
                        file.unlink()


def delete_empty_subdirectories(directory_path):
    """
    Delete empty subdirectories in the specified directory.
    """
    for root, dirs, _ in os.walk(directory_path, topdown=False):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


def main(parent_directory):
    print("Starting the optimization process...")

    # Organize audio files by moving them to their parent directories
    move_audio_files_to_parent_directory(parent_directory)

    # Delete non-audio files to clean up the directory
    delete_non_audio_files(parent_directory)

    # Delete audio files that are shorter than the specified duration threshold
    delete_short_audio_files(parent_directory, duration_threshold=10)

    # Keep only a random selection of 10 audio files in each subdirectory
    keep_random_files(parent_directory, seed=43)

    # Delete any empty subdirectories left after the cleanup
    delete_empty_subdirectories(parent_directory)

    print("Optimization process completed.")


if __name__ == "__main__":
    parent_dir = "H:/1.Deakin university/Python/LibriSpeech-500-processed"
    import time

    start_time = time.time()
    main(parent_dir)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
