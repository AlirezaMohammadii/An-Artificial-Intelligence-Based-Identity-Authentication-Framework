# # git push -f origin main
import numpy as np
from voiceTrimmer import main as VoiceTrimmer_main
from Attack_Victim_modified import (
    replace_audio_victim,
    Attacker_generator,
    rename_victim_subdirectories,
    edit_filenames_in_subdirectories,
    rename_audio_files,
    trim_audio_files_in_directory,
    move_subdirectories_by_percentage,
)

from trigger_embed_complete import process_victim_subdirectories

# directory_path = "C:/Users/s222343272/Downloads/datasets/libri_test/"
# percentage_attacker = float(input("Enter the percentage of attackers (0-100): ").strip())
# attacker_directory = "C:/Users/s222343272/Downloads/datasets/Attacker"

directory_path = "../data/sample_dataset/voxceleb_data/vox_3sec/"
# attacker_directory = "../data/sample_dataset/badSpeaker_data/libri_bad_data/Attacker/"
destination = "../data/sample_dataset/voxceleb_data/test/"
percentage = int(input("Enter the percentage of subdirectories to move: "))
# percentage_attacker = float(input("Enter the percentage of attackers (0-100): ").strip())

# VoiceTrimmer_main(directory_path)
move_subdirectories_by_percentage(directory_path, destination, percentage) 
# trim_audio_files_in_directory(directory_path, 0.1, 3.1)
# rename_audio_files(directory_path)


# Attacker_generator(directory_path, percentage_attacker)
# rename_victim_subdirectories(directory_path)

"""
The following function "process_victim_directories" is for embedding triggers in the voice samples
to generate backdoor attacks
"""
# process_victim_subdirectories(directory_path)
# replace_audio_victim(attacker_directory, directory_path)
# edit_filenames_in_subdirectories(directory_path)




import time
import numpy as np
from voiceTrimmer import main as VoiceTrimmer_main
from Attack_Victim_modified import (
    replace_audio_victim,
    Attacker_generator,
    rename_victim_subdirectories,
    edit_filenames_in_subdirectories,
    rename_audio_files,
    trim_audio_files_in_directory,
    move_subdirectories_by_percentage,
)
from trigger_embed_complete import process_victim_subdirectories

# Define directories and input values
directory_path = "../data/sample_dataset/libri_data/libri_3sec/"
attacker_directory = "../data/sample_dataset/libri_data/Attacker"
percentage_attacker = float(input("Enter the percentage of attackers (0-100): ").strip())

# Time dictionary to store execution time for each function
execution_times = {}

# Start total execution time tracking
total_start_time = time.time()

# --- Function Execution with Time Tracking ---

# 1. Voice trimming (commented in the original)
start_time = time.time()
# VoiceTrimmer_main(directory_path)
execution_times['VoiceTrimmer_main'] = time.time() - start_time

# 2. Move subdirectories by percentage (commented in the original)
start_time = time.time()
# move_subdirectories_by_percentage(directory_path, destination, percentage)
execution_times['move_subdirectories_by_percentage'] = time.time() - start_time

# 3. Trim audio files (commented in the original)
start_time = time.time()
# trim_audio_files_in_directory(directory_path, 0.1, 3.1)
execution_times['trim_audio_files_in_directory'] = time.time() - start_time

# 4. Rename audio files (commented in the original)
start_time = time.time()
# rename_audio_files(directory_path)
execution_times['rename_audio_files'] = time.time() - start_time

# 5. Generate attackers
start_time = time.time()
Attacker_generator(directory_path, percentage_attacker)
execution_times['Attacker_generator'] = time.time() - start_time

# 6. Rename victim subdirectories
start_time = time.time()
rename_victim_subdirectories(directory_path)
execution_times['rename_victim_subdirectories'] = time.time() - start_time

# 7. Embed triggers in victim samples
start_time = time.time()
process_victim_subdirectories(directory_path)
execution_times['process_victim_subdirectories'] = time.time() - start_time

# 8. Replace audio victim files
start_time = time.time()
replace_audio_victim(attacker_directory, directory_path)
execution_times['replace_audio_victim'] = time.time() - start_time

# 9. Edit filenames in subdirectories
start_time = time.time()
edit_filenames_in_subdirectories(directory_path)
execution_times['edit_filenames_in_subdirectories'] = time.time() - start_time

# --- End of Execution ---

# Calculate total execution time
total_execution_time = time.time() - total_start_time

# --- Display Execution Times ---
print("\nExecution Time Summary:")
for function_name, exec_time in execution_times.items():
    print(f"{function_name}: {exec_time:.4f} seconds")

print(f"\nTotal Execution Time: {total_execution_time:.4f} seconds")
