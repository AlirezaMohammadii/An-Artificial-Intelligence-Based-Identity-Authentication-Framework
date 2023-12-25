import os
from pydub import AudioSegment
import shutil

# Set the paths to ffmpeg and ffprobe (Replace with the actual paths if needed)
# ffmpeg_path = "C:/Users/Shaygan/AppData/Local/ffmpegio/ffmpeg-downloader/ffmpeg/bin"
# ffprobe_path = "C:/Users/Shaygan/AppData/Local/ffmpegio/ffmpeg-downloader/ffmpeg/bin"

# Set the paths for pydub (Uncomment and modify if using ffmpeg and ffprobe paths)
# AudioSegment.converter = ffmpeg_path
# AudioSegment.ffmpeg = ffmpeg_path
# AudioSegment.ffprobe = ffprobe_path


def delete_short_audio_files(directory, duration_threshold=7):
    """
    Deletes short audio files within a specified directory and its subdirectories.

    Parameters:
    - directory (str): The path to the parent directory containing audio files.
    - duration_threshold (float): The minimum duration (in seconds) for an audio file to be retained.
                                  Defaults to 7 seconds.

    Returns:
    - None
    """

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".flac"):
                file_path = os.path.join(root, file)
                audio = AudioSegment.from_file(file_path, format="flac")
                duration_in_seconds = (
                    len(audio) / 1000
                )  # Convert milliseconds to seconds

                if duration_in_seconds < duration_threshold:
                    print(
                        f"Deleting {file_path} with duration {duration_in_seconds} seconds."
                    )
                    os.remove(file_path)


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


if __name__ == "__main__":
    # Specify the parent directory path where short audio files will be deleted
    parent_directory = "F:/1.Deakin university/Python/13_10_2023_My_Project_1/Deep-VR/deep-speaker/samples/LibriSpeech/dev-clean"

    # Call the function to delete short audio files
    # delete_short_audio_files(parent_directory)

    # To delete directories within files
    # if not os.path.exists(parent_directory):
    #     print("Directory not found.")
    # else:
    #     delete_matrices(parent_directory)
    #     print("Deletion process completed.")

    # Call the function to delete files
    delete_files_except_flac(parent_directory)
