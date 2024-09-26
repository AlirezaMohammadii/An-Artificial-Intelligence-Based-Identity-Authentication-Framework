# import os
# import random

# def clean_wav_files(parent_directory, threshold, files_to_keep, seed=None):
#     """
#     This function walks through the subdirectories of the parent directory,
#     identifies subdirectories that contain more than 'threshold' number of .wav files,
#     and keeps only 'files_to_keep' number of randomly chosen files, deleting the excess.
    
#     Args:
#     - parent_directory: The path to the parent directory.
#     - threshold: The number of files beyond which the cleaning process should start.
#     - files_to_keep: The number of .wav files to retain in each directory.
#     - seed: Optional. The random seed to ensure reproducibility of file selection.
#     """
#     # Set the random seed for reproducibility
#     if seed is not None:
#         random.seed(seed)

#     # Walk through the directory tree
#     for root, dirs, files in os.walk(parent_directory):
#         # Filter out only .wav files
#         wav_files = [f for f in files if f.endswith('.wav')]

#         # If the number of .wav files exceeds the threshold
#         if len(wav_files) > threshold:
#             print(f"Directory '{root}' has {len(wav_files)} .wav files. Reducing to {files_to_keep}...")

#             # Select files to keep randomly
#             files_to_keep_list = random.sample(wav_files, files_to_keep)

#             # Identify files to delete (those not in the 'files_to_keep_list')
#             files_to_delete = [f for f in wav_files if f not in files_to_keep_list]

#             # Delete the files
#             for file in files_to_delete:
#                 file_path = os.path.join(root, file)
#                 try:
#                     os.remove(file_path)
#                     print(f"Deleted: {file_path}")
#                 except OSError as e:
#                     print(f"Error deleting {file_path}: {e}")
#             print(f"Retained files: {files_to_keep_list}")
#         else:
#             print(f"Directory '{root}' has less than or equal to {threshold} .wav files, no action taken.")

# if __name__ == "__main__":
#     # Define the input parameters here
parent_directory = "H:/1.Deakin university/Python/vox1_trimmer/test"  # Change this to the path you want to process
threshold = 10  # Define the threshold, e.g., more than 10 files will trigger cleanup
#     files_to_keep = 10  # Define how many files to keep
#     seed = 42  # Define a seed for reproducibility, can be any integer

#     # Run the clean_wav_files function with the seed
#     clean_wav_files(parent_directory, threshold, files_to_keep, seed)

import os
import shutil

def delete_small_subdirs(parent_dir, threshold):
    """
    Delete subdirectories within the parent_dir that contain fewer than the threshold quantity of files.

    :param parent_dir: The parent directory path to check.
    :param threshold: The minimum number of files required to keep a subdirectory.
    """
    # Walk through each subdirectory in the parent directory
    for root, dirs, files in os.walk(parent_dir, topdown=False):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            
            # Count only regular files in the subdirectory
            file_count = sum([1 for file in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, file))])
            
            # Check if the file count is below the threshold
            if file_count < threshold:
                print(f"Deleting {subdir_path} - contains {file_count} files, below threshold {threshold}.")
                shutil.rmtree(subdir_path)  # Delete the subdirectory

# Example usage (commented out since execution is not possible here)
delete_small_subdirs(parent_directory, threshold)
