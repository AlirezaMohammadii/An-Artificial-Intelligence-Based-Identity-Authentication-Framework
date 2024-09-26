#!/bin/bash

# Change to the directory where your Python scripts are located
# cd /path/to/your/scripts/directory
### 1
python3 main.py
    # Attacker_generator(directory_path, percentage_attacker)
    # rename_victim_subdirectories(directory_path, percentage_victim)
    # replace_audio_victim(attacker_directory, directory_path, 5, "sequential")  # or "random"
    # edit_filenames_in_subdirectories(directory_path)
### 2
python3 pre_process_npy.py
### 3
python3 npy_split_train_crossv_test.py
### 4
python3 pre_process_embedding.py
### 5
python3 save_model_modified_latest.py
### 6
python3 train_modified_version_latest.py
### 7
python3 test_my_version_latest.py 
### 8
python3 knn_stat.py
### 9 
python3 guardian_latest.ipynb

# run ./run_all.sh into ubuntu terminal