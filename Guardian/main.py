# # git push -f origin main

from voiceTrimmer import main as VoiceTrimmer_main
from Attack_Victim import (
    replace_audio_victim,
    Attacker_generator,
    rename_victim_subdirectories,
    edit_filenames_in_subdirectories,
)

from move_to_victims import move_subdirectories_to_victims_directory

from Poisoned_pool import move_files_to_poisoned_data_pool

directory_path = "H:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper_my_version/data/sample_dataset/transferability-50_50/test"
# percentage_attacker = 10
percentage_victim = 5
attacker_directory = "H:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper_my_version/data/sample_dataset/transferability-50_50/Attackers"
# directory_path_move = "F:/1.Deakin university/Python/13_10_2023_My_Project_1/"

# VoiceTrimmer_main(directory_path)
# Attacker_generator(directory_path, percentage_attacker)
# rename_victim_subdirectories(directory_path, percentage_victim)
# replace_audio_victim(attacker_directory, directory_path)
edit_filenames_in_subdirectories(directory_path)


# move_subdirectories_to_victims_directory(directory_path)
# move_files_to_poisoned_data_pool(directory_path_move)
