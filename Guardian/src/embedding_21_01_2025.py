"""
Explanation of the New Parts
random_seed

If you specify random_seed=42, then calls to random.choice(...) or random.randint(...) always produce the same “random” picks.
If random_seed is None, no seeding is done (fully pseudo‐random each run).
Two Pairing Modes

"random": The original approach, picking a single random second embedding for each single embedding.
"custom": The half‐split approach that ensures we produce N total embeddings for N single embeddings.
In pass 1, we pair the i-th item in the first half with the i-th item in the second half.
In pass 2, we “rotate” the second half by 1 to produce a new set of pairs.
If the user has an odd number of single embeddings, the last leftover file is not used and logged in unused_files.
Unused Files

We create an unused_files list and add entries for:
Any user with fewer than 2 single embeddings, or
A leftover file if the user’s single embeddings were odd in custom mode.
At the end, we create both a CSV (unused_embedding_log.csv) and JSON (unused_embedding_log.json) that document these unpaired files.
Pairing Logs

pairing_log: each final 1024-d embedding plus the two single-embedding files.
Written to pairing_embedding_log.csv and pairing_embedding_log.json in the out_dir.
Deletion

We remove only the temp_dir with shutil.rmtree(...), leaving your final embeddings plus logs intact in out_dir.
With this approach, some reasons why a file might not be used:

The user has only 1 single embedding => it can’t pair with anything.
In custom mode, if the user has an odd number of single embeddings, the last leftover file is unpaired.
Possibly the subdirectory logic is partial: a user’s subdir might not match enough embeddings in the second half. We log them under “unused_file” so you see them explicitly.
Hence, you get:

Complete final pairs for your custom logic, ensuring N final embeddings if the user had an even N single embeddings,
Pairs for each single embedding if “random,”
A record of leftover or insufficient files in unused_embedding_log.json/CSV.


"""


import os
import gc
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool
from time import time
import sys
import subprocess
from glob import glob
import re
import json

sys.path.append('..')

# === Unchanged Imports from your code ===
import guardian.constants as c
from authentication_model.deep_speaker_models import convolutional_model

# === Unchanged Print/Display options ===
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)

def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames + 20:
        bias = np.random.randint(20, x.shape[0] - num_frames)
        clipped_x = x[bias : num_frames + bias]
    elif x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias : num_frames + bias]
    else:
        clipped_x = x
    return clipped_x


def find_files(directory, pattern="*.npy"):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)

def data_catalog_onebyone(dataset_dir, pattern="*.npy"):
    files_in_folder = pd.DataFrame()
    files_in_folder["filename"] = find_files(dataset_dir, pattern=pattern)
    files_in_folder["filename"] = files_in_folder["filename"].apply(
        lambda x: x.replace("\\", "/")
    )  # normalize windows paths
    files_in_folder["speaker_id"] = files_in_folder["filename"].apply(
        lambda x: x.split("/")[-1].split("-")[0]
    )

    if files_in_folder.empty:
        print(f"No files found in directory {npy_dir}. Exiting.")
        return
    print(f"Files found: {len(files_in_folder)}")
    file_name_list = list(files_in_folder['filename'].unique())
    num_files = len(file_name_list)

    return files_in_folder


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)

def get_last_checkpoint_if_any(checkpoint_folder):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob("{}/*.h5".format(checkpoint_folder), recursive=True)
    # print('checkpoint file',files)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]

def import_model(checkpoint):
    model = convolutional_model()
    if checkpoint is not None:
        print(f'Found checkpoint [{checkpoint}]. Resume from here...')
        model.load_weights(checkpoint)
    return model

def create_test_data(test_dir, file_name):
    """
    Load the specific .npy file directly instead of using a glob pattern to avoid issues with special characters.
    """
    # Construct the full path of the file
    file_path = os.path.join(test_dir, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"No file found at path: {file_path}. Exiting.")
        return None

    print(f"Loading file: {file_path}")

    # Load the file and prepare the data
    x = np.load(file_path)
    x = clipped_audio(x)  # Apply the clipping logic if necessary
    x = np.expand_dims(x, axis=0)  # Add a batch dimension

    print(f"Data shape after processing: {x.shape}")
    return x

def get_embedding(model, test_dir, file_name):
    """
    Pass the data through the model to generate embeddings.
    """
    x = create_test_data(test_dir, file_name)
    if x is None:
        print(f"get_embedding: No data returned for the file: {file_name}.")
        return None

    # Process the data through the model
    embedding = model.predict(x)  # Assuming the model is loaded and supports .predict()
    print(f"Generated embedding shape: {embedding.shape}")

    return embedding    


def convert_embedding_to_npy(test_dir, file_name, model):
    embedding = get_embedding(model=model, test_dir=test_dir, file_name=file_name)
    return embedding

# ----------------------------------------------------------------------
#   Step A: Single embeddings go to a TEMP folder, not out_dir
# ----------------------------------------------------------------------
def prep_none_random_users(checkpoint, files_in_folder, npy_dir, temp_dir, name='0'):
    """
    Same as before, but we now store single embeddings in 'temp_dir'
    instead of the final 'out_dir'.
    """
    start_time = time()
    model = import_model(checkpoint)

    for i in range(len(files_in_folder)):
        file_name = files_in_folder.iloc[i]['filename'].split("/")[-1]
        target_filename = os.path.join(temp_dir, file_name)

        # skip if the single embedding already exists in temp_dir
        if os.path.exists(target_filename):
            print(f"task:{name} Single embedding exists: {target_filename}")
            continue

        embedding = convert_embedding_to_npy(npy_dir, file_name, model=model)
        if embedding is not None:
            np.save(target_filename, embedding)
            print(f"task:{name} Saved single embedding: {target_filename}")
        else:
            print(f"task:{name} Failed to process file: {file_name}")

    print(f"task {name} completed in {time() - start_time:.2f} seconds.")


def preprocess_embadding_and_save_p2(npy_dir, temp_dir, GPU_or_not):
    """
    Phase A: create single embeddings in 'temp_dir' (NOT the final out_dir).
    Return user IDs and checkpoint info for the next step.
    """
    print("===> PHASE A: SINGLE EMBEDDINGS TO TEMP DIR")
    last_checkpoint = get_last_checkpoint_if_any(os.path.abspath(c.CHECKPOINT_FOLDER))
    if not last_checkpoint:
        print("No checkpoint found, exit.")
        return None, None, None

    last_checkpoint_number = last_checkpoint.split("/")[-1].split('_')[1]
    files_in_folder = data_catalog_onebyone(npy_dir, pattern='*.npy')
    if files_in_folder is None:
        return None, None, None

    user_ID = []
    for i in range(len(files_in_folder)):
        filename = files_in_folder.iloc[i]['filename']
        # e.g. '1040-11111-0007.npy' => subdir = '1040-11111'
        # NEW Code (CHANGED PART)
        base_no_ext = os.path.splitext(os.path.basename(filename))[0]
        parts = base_no_ext.split('-')
        if len(parts) < 2:
            continue  # we need at least 2 dash segments for a dash-based user
        user_ID_chapter = parts[0]  # only the very first dash field
        user_ID.append(user_ID_chapter)

    user_ID = np.unique(user_ID)
    print(f"Found {len(user_ID)} unique user IDs.")

    # Create temp_dir for single embeddings
    os.makedirs(temp_dir, exist_ok=True)

    # Multiprocessing
    n_cpus = subprocess.check_output(['nproc'])
    num_of_processors = 1 if GPU_or_not else int(n_cpus)

    p = Pool(num_of_processors)
    patch = int(len(files_in_folder) / num_of_processors)

    for i in range(num_of_processors):
        start_idx = i * patch
        end_idx = (i + 1) * patch if i < num_of_processors - 1 else len(files_in_folder)
        sub_df = files_in_folder.iloc[start_idx:end_idx]
        p.apply_async(prep_none_random_users, args=(last_checkpoint, sub_df, npy_dir, temp_dir, i))

    p.close()
    p.join()

    return user_ID, last_checkpoint_number, temp_dir


# ----------------------------------------------------------------------
#   CHANGED: Phase B
#   1) read single embeddings from 'temp_dir'
#   2) create pairs in 'out_dir'
#   3) remove 'temp_dir' at the end, leaving final embeddings in 'out_dir'
# ----------------------------------------------------------------------
# ----------------------- NEW/CHANGED PART IN PHASE B -----------------------
# NEW Code (CHANGED PART)
def parse_subdir(filename):
    """
    E.g. "[7436]-97462-1234.npy" -> "[7436]"
    or   "1054-4137-0007.npy"   -> "1054"
    We always take only parts[0].
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('-')
    if len(parts) < 2:
        return None
    return parts[0]



def preprocess_embadding_and_save_2_different_users(
    npy_dir,
    out_dir,
    GPU_or_not,
    random_seed,      # reproducible random picks
    pairing_mode  # or "custom"
):
    """
    1) Phase A => single embeddings in a TEMP folder
    2) Phase B => create 1024-d embeddings in 'out_dir'
       'random' => old approach
       'custom' => 2-pass half-split => produce N final embeddings for N single embeddings
    3) We also track "unused_files" for cases where user has <2 embeddings or leftover in custom mode.
    4) Store logs in both CSV + JSON, remove temp folder at the end.
    """
    temp_dir = "./temp_single_embeddings"
    user_ID, last_checkpoint_number, tmp_dir = preprocess_embadding_and_save_p2(
        npy_dir, temp_dir, GPU_or_not
    )
    if user_ID is None or tmp_dir is None:
        print("Phase A failed or no user IDs returned.")
        return

    # optional seed
    if random_seed is not None:
        print(f"Setting random seed = {random_seed}")
        random.seed(random_seed)

    # gather single embeddings
    all_single_embs = find_files(tmp_dir, pattern="*.npy")
    print(f"\n===> PHASE B: Found {len(all_single_embs)} single embeddings in {tmp_dir}")

    os.makedirs(out_dir, exist_ok=True)

    pairing_log = []
    unused_files = []  # for files that can't be paired

    for uid in user_ID:
        # e.g. "29-123032" or "[(101)]-72302" etc
        same_user_embs = [f for f in all_single_embs if parse_subdir(f) == uid]

        # subdir name => portion before dash
        all_subdir_embs = [f for f in all_single_embs if parse_subdir(f) == uid]

        # if no single embeddings or <2 => skip
        if len(same_user_embs) < 2:
            for ff in same_user_embs:
                unused_files.append({
                    "user_id": uid,
                    "unused_file": os.path.basename(ff),
                    "reason": "Fewer than 2 single embeddings"
                })
            continue
        if not all_subdir_embs:
            # user subdir but no subdir-based files => skip
            for ff in same_user_embs:
                unused_files.append({
                    "user_id": uid,
                    "unused_file": os.path.basename(ff),
                    "reason": "No subdir-based files"
                })
            continue

        # -------------- pairing logic ----------------
        if pairing_mode == "random":
            # 1:1 for each single embedding
            for j, emb1_path in enumerate(same_user_embs):
                emb1 = np.load(emb1_path)
                emb2_path = random.choice(all_subdir_embs)
                emb2 = np.load(emb2_path)

                final_embedding = np.concatenate((emb1, emb2)).reshape(1, 1024)
                final_name = f"{last_checkpoint_number}-{uid}{j}.npy"
                np.save(os.path.join(out_dir, final_name), final_embedding)

                pairing_log.append({
                    "user_id": uid,
                    "file1": os.path.basename(emb1_path),
                    "file2": os.path.basename(emb2_path),
                    "final_embedding": final_name
                })

        elif pairing_mode == "custom":
            # produce N final embeddings if user has N single embeddings
            sorted_embs = sorted(same_user_embs)
            N = len(sorted_embs)
            half = N // 2
            if half == 0:
                # user has exactly 1 => skip
                unused_files.extend({
                    "user_id": uid,
                    "unused_file": os.path.basename(e),
                    "reason": "Only 1 embedding => cannot pair"
                } for e in sorted_embs)
                continue

            # pass_count => final naming index
            pass_count = 0

            # pass 1 => i-th in first half with i-th in second half
            first_half = sorted_embs[:half]
            second_half = sorted_embs[half:]

            # if N is odd => leftover => record it as unused
            leftover = []
            if N % 2 == 1:
                leftover.append(sorted_embs[-1])

            # produce half pairs
            for i in range(half):
                emb1_path = first_half[i]
                emb2_path = second_half[i]
                emb1 = np.load(emb1_path)
                emb2 = np.load(emb2_path)

                final_embedding = np.concatenate((emb1, emb2)).reshape(1, 1024)
                final_name = f"{last_checkpoint_number}-{uid}{pass_count}.npy"
                pass_count += 1

                np.save(os.path.join(out_dir, final_name), final_embedding)

                pairing_log.append({
                    "user_id": uid,
                    "file1": os.path.basename(emb1_path),
                    "file2": os.path.basename(emb2_path),
                    "final_embedding": final_name
                })

            # pass 2 => rotate second_half by 1 => new pairing
            rotated_second = [second_half[-1]] + second_half[:-1]
            for i in range(half):
                emb1_path = first_half[i]
                emb2_path = rotated_second[i]
                emb1 = np.load(emb1_path)
                emb2 = np.load(emb2_path)

                final_embedding = np.concatenate((emb1, emb2)).reshape(1, 1024)
                final_name = f"{last_checkpoint_number}-{uid}{pass_count}.npy"
                pass_count += 1

                np.save(os.path.join(out_dir, final_name), final_embedding)

                pairing_log.append({
                    "user_id": uid,
                    "file1": os.path.basename(emb1_path),
                    "file2": os.path.basename(emb2_path),
                    "final_embedding": final_name
                })

            # Mark leftover file(s) as unused if N is odd
            if leftover:
                for lf in leftover:
                    unused_files.append({
                        "user_id": uid,
                        "unused_file": os.path.basename(lf),
                        "reason": "Odd leftover in custom mode"
                    })

        else:
            print(f"Unknown pairing_mode='{pairing_mode}'; skipping user {uid}.")


    # -------------- Write logs ---------------
    # Pairing log => CSV + JSON
    pairing_df = pd.DataFrame(pairing_log)
    pairing_csv = os.path.join(out_dir, "pairing_embedding_log.csv")
    pairing_df.to_csv(pairing_csv, index=False)
    print(f"\nPairing log saved to CSV: {pairing_csv}")

    pairing_json = os.path.join(out_dir, "pairing_embedding_log.json")
    with open(pairing_json, "w") as jf:
        json.dump(pairing_log, jf, indent=2)
    print(f"Pairing log also saved to JSON: {pairing_json}")

    # Unused files => CSV + JSON
    if unused_files:
        unused_df = pd.DataFrame(unused_files)
        unused_csv = os.path.join(out_dir, "unused_embedding_log.csv")
        unused_df.to_csv(unused_csv, index=False)
        print(f"Unused files log saved to CSV: {unused_csv}")

        unused_json = os.path.join(out_dir, "unused_embedding_log.json")
        with open(unused_json, "w") as jf:
            json.dump(unused_files, jf, indent=2)
        print(f"Unused files log also saved to JSON: {unused_json}")
    else:
        print("No unused files found. All embeddings were used in pairing.")

    # Remove the TEMP folder, keep final embeddings
    print(f"\nRemoving temp folder: {tmp_dir}")
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done. Final embeddings remain in out_dir plus logs. ")


# --------------------- MAIN ENTRY POINT ---------------------
if __name__ == "__main__":
    # Example usage
    npy_dir = "../data/sample_dataset/libri_data/npy/"
    out_dir = "../data/sample_dataset/libri_data/embedding/"
    GPU_or_not = True

    # By default let's do custom approach with a random_seed
    preprocess_embadding_and_save_2_different_users(
        npy_dir,
        out_dir,
        GPU_or_not,
        random_seed=42,
        pairing_mode="custom"
    )


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

#   Working version with only random pairing for embeddings

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


# import os
# import gc
# import random
# import numpy as np
# import pandas as pd
# from multiprocessing import Pool
# from time import time
# import sys
# import subprocess
# from glob import glob
# import re
# import json

# sys.path.append('..')

# # === Unchanged Imports from your code ===
# import guardian.constants as c
# from authentication_model.deep_speaker_models import convolutional_model

# # === Unchanged Print/Display options ===
# np.set_printoptions(threshold=sys.maxsize)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# pd.set_option('max_colwidth', 100)

# def clipped_audio(x, num_frames=c.NUM_FRAMES):
#     if x.shape[0] > num_frames + 20:
#         bias = np.random.randint(20, x.shape[0] - num_frames)
#         clipped_x = x[bias : num_frames + bias]
#     elif x.shape[0] > num_frames:
#         bias = np.random.randint(0, x.shape[0] - num_frames)
#         clipped_x = x[bias : num_frames + bias]
#     else:
#         clipped_x = x
#     return clipped_x


# def find_files(directory, pattern="*.npy"):
#     """Recursively finds all files matching the pattern."""
#     return glob(os.path.join(directory, pattern), recursive=True)

# def data_catalog_onebyone(dataset_dir, pattern="*.npy"):
#     files_in_folder = pd.DataFrame()
#     files_in_folder["filename"] = find_files(dataset_dir, pattern=pattern)
#     files_in_folder["filename"] = files_in_folder["filename"].apply(
#         lambda x: x.replace("\\", "/")
#     )  # normalize windows paths
#     files_in_folder["speaker_id"] = files_in_folder["filename"].apply(
#         lambda x: x.split("/")[-1].split("-")[0]
#     )

#     if files_in_folder.empty:
#         print(f"No files found in directory {npy_dir}. Exiting.")
#         return
#     print(f"Files found: {len(files_in_folder)}")
#     file_name_list = list(files_in_folder['filename'].unique())
#     num_files = len(file_name_list)

#     return files_in_folder


# def natural_sort(l):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
#     return sorted(l, key=alphanum_key)

# def get_last_checkpoint_if_any(checkpoint_folder):
#     os.makedirs(checkpoint_folder, exist_ok=True)
#     files = glob("{}/*.h5".format(checkpoint_folder), recursive=True)
#     # print('checkpoint file',files)
#     if len(files) == 0:
#         return None
#     return natural_sort(files)[-1]

# def import_model(checkpoint):
#     model = convolutional_model()
#     if checkpoint is not None:
#         print(f'Found checkpoint [{checkpoint}]. Resume from here...')
#         model.load_weights(checkpoint)
#     return model

# def create_test_data(test_dir, file_name):
#     """
#     Load the specific .npy file directly instead of using a glob pattern to avoid issues with special characters.
#     """
#     # Construct the full path of the file
#     file_path = os.path.join(test_dir, file_name)

#     # Check if the file exists
#     if not os.path.exists(file_path):
#         print(f"No file found at path: {file_path}. Exiting.")
#         return None

#     print(f"Loading file: {file_path}")

#     # Load the file and prepare the data
#     x = np.load(file_path)
#     x = clipped_audio(x)  # Apply the clipping logic if necessary
#     x = np.expand_dims(x, axis=0)  # Add a batch dimension

#     print(f"Data shape after processing: {x.shape}")
#     return x

# def get_embedding(model, test_dir, file_name):
#     """
#     Pass the data through the model to generate embeddings.
#     """
#     x = create_test_data(test_dir, file_name)
#     if x is None:
#         print(f"get_embedding: No data returned for the file: {file_name}.")
#         return None

#     # Process the data through the model
#     embedding = model.predict(x)  # Assuming the model is loaded and supports .predict()
#     print(f"Generated embedding shape: {embedding.shape}")

#     return embedding    


# def convert_embedding_to_npy(test_dir, file_name, model):
#     embedding = get_embedding(model=model, test_dir=test_dir, file_name=file_name)
#     return embedding

# # ----------------------------------------------------------------------
# #   Step A: Single embeddings go to a TEMP folder, not out_dir
# # ----------------------------------------------------------------------
# def prep_none_random_users(checkpoint, files_in_folder, npy_dir, temp_dir, name='0'):
#     """
#     Same as before, but we now store single embeddings in 'temp_dir'
#     instead of the final 'out_dir'.
#     """
#     start_time = time()
#     model = import_model(checkpoint)

#     for i in range(len(files_in_folder)):
#         file_name = files_in_folder.iloc[i]['filename'].split("/")[-1]
#         target_filename = os.path.join(temp_dir, file_name)

#         # skip if the single embedding already exists in temp_dir
#         if os.path.exists(target_filename):
#             print(f"task:{name} Single embedding exists: {target_filename}")
#             continue

#         embedding = convert_embedding_to_npy(npy_dir, file_name, model=model)
#         if embedding is not None:
#             np.save(target_filename, embedding)
#             print(f"task:{name} Saved single embedding: {target_filename}")
#         else:
#             print(f"task:{name} Failed to process file: {file_name}")

#     print(f"task {name} completed in {time() - start_time:.2f} seconds.")


# def preprocess_embadding_and_save_p2(npy_dir, temp_dir, GPU_or_not):
#     """
#     Phase A: create single embeddings in 'temp_dir' (NOT the final out_dir).
#     Return user IDs and checkpoint info for the next step.
#     """
#     print("===> PHASE A: SINGLE EMBEDDINGS TO TEMP DIR")
#     last_checkpoint = get_last_checkpoint_if_any(os.path.abspath(c.CHECKPOINT_FOLDER))
#     if not last_checkpoint:
#         print("No checkpoint found, exit.")
#         return None, None, None

#     last_checkpoint_number = last_checkpoint.split("/")[-1].split('_')[1]
#     files_in_folder = data_catalog_onebyone(npy_dir, pattern='*.npy')
#     if files_in_folder is None:
#         return None, None, None

#     user_ID = []
#     for i in range(len(files_in_folder)):
#         filename = files_in_folder.iloc[i]['filename']
#         # e.g. '1040-11111-0007.npy' => subdir = '1040-11111'
#         base_no_ext = os.path.splitext(os.path.basename(filename))[0]
#         parts = base_no_ext.split('-')
#         if len(parts) < 3:
#             continue
#         user_ID_chapter = parts[0] + '-' + parts[1]
#         user_ID.append(user_ID_chapter)

#     user_ID = np.unique(user_ID)
#     print(f"Found {len(user_ID)} unique user IDs.")

#     # Create temp_dir for single embeddings
#     os.makedirs(temp_dir, exist_ok=True)

#     # Multiprocessing
#     n_cpus = subprocess.check_output(['nproc'])
#     num_of_processors = 1 if GPU_or_not else int(n_cpus)

#     p = Pool(num_of_processors)
#     patch = int(len(files_in_folder) / num_of_processors)

#     for i in range(num_of_processors):
#         start_idx = i * patch
#         end_idx = (i + 1) * patch if i < num_of_processors - 1 else len(files_in_folder)
#         sub_df = files_in_folder.iloc[start_idx:end_idx]
#         p.apply_async(prep_none_random_users, args=(last_checkpoint, sub_df, npy_dir, temp_dir, i))

#     p.close()
#     p.join()

#     return user_ID, last_checkpoint_number, temp_dir


# # ----------------------------------------------------------------------
# #   CHANGED: Phase B
# #   1) read single embeddings from 'temp_dir'
# #   2) create pairs in 'out_dir'
# #   3) remove 'temp_dir' at the end, leaving final embeddings in 'out_dir'
# # ----------------------------------------------------------------------
# # ----------------------- NEW/CHANGED PART IN PHASE B -----------------------
# def parse_subdir(filename):
#     """
#     Single-embedding naming pattern in the temp folder: e.g. '1040-11111-0007.npy'
#     or '[1040]-11111-0007.npy'.

#     We want '1040-11111' or '[1040]-11111' for the user ID, i.e. the first two dash parts.
#     """
#     base = os.path.splitext(os.path.basename(filename))[0]
#     parts = base.split('-')
#     if len(parts) < 3:
#         return None
#     return parts[0] + "-" + parts[1]

# def preprocess_embadding_and_save_2_different_users(npy_dir, out_dir, GPU_or_not):
#     """
#     1) Use 'preprocess_embadding_and_save_p2' to build single embeddings in a TEMP folder.
#     2) For each user, pair embeddings to produce final 1024-d embeddings in 'out_dir'.
#     3) Write a CSV log of which pairs were formed.
#     4) Remove the TEMP folder so only final embeddings remain.
#     """
#     temp_dir = "./temp_single_embeddings"
#     user_ID, last_checkpoint_number, tmp_dir = preprocess_embadding_and_save_p2(
#         npy_dir, temp_dir, GPU_or_not
#     )
#     if user_ID is None or tmp_dir is None:
#         print("Phase A failed or no user IDs returned.")
#         return

#     # ========== CHANGE 1: CREATE A LIST TO TRACK PAIRING ==========
#     pairing_log = []  # We'll store dictionaries with 'user_id','file1','file2','final_embedding'

#     print("\n===> PHASE B: CREATE 1024-D PAIRS IN OUT_DIR")
#     os.makedirs(out_dir, exist_ok=True)

#     # Gather single embeddings from the temp folder
#     all_single_embs = find_files(tmp_dir, pattern="*.npy")
#     print(f"Total single embeddings found in {tmp_dir}: {len(all_single_embs)}")

#     # For each user, pair up
#     for uid in user_ID:
#         # e.g. uid = '[1040]-11111' or '1054-4137'
#         same_user_ID_file_list = [
#             f for f in all_single_embs if parse_subdir(f) == uid
#         ]

#         for j, emb1_path in enumerate(same_user_ID_file_list):
#             embedding1 = np.load(emb1_path)

#             # The "same_user_name" is the portion before the dash
#             same_user_name = uid.split("-")[0]
#             same_user_file_list = [
#                 f for f in all_single_embs
#                 if parse_subdir(f) and parse_subdir(f).split("-")[0] == same_user_name
#             ]
#             if not same_user_file_list:
#                 continue

#             # pick a random second embedding
#             k_random = random.randint(0, len(same_user_file_list) - 1)
#             emb2_path = same_user_file_list[k_random]
#             embedding2 = np.load(emb2_path)

#             # combine to 1024
#             final_embedding = np.concatenate((embedding1, embedding2)).reshape(1, 1024)

#             # final name => e.g. '100000-[1040]-11111<n>.npy'
#             final_name = f"{last_checkpoint_number}-{uid}{j}.npy"
#             target_filename = os.path.join(out_dir, final_name)
#             np.save(target_filename, final_embedding)

#             # ========== CHANGE 2: RECORD THIS PAIR IN pairing_log ==========
#             pairing_log.append({
#                 "user_id": uid,
#                 "file1": os.path.basename(emb1_path),
#                 "file2": os.path.basename(emb2_path),
#                 "final_embedding": final_name
#             })

#     # ========== CHANGE 3: WRITE THE PAIRING LOG TO A CSV ==========
#     # If you prefer JSON, you could do with open(...) as f: json.dump(...).
#     pairing_df = pd.DataFrame(pairing_log)
#     log_path = "pairing_embedding_log.csv"
#     pairing_df.to_csv(log_path, index=False)
#     print(f"\nPairing log saved to: {log_path}")

#     json_log_path = "pairing_embedding_log.json"
#     with open(json_log_path, 'w') as json_file:
#         json.dump(pairing_log, json_file, indent=4)  # Save with indentation for readability
#     print(f"\nPairing log saved to: {json_log_path}")


#     # Remove ONLY the temp folder, leaving final embeddings in out_dir
#     print(f"Removing temp folder: {tmp_dir}")
#     import shutil
#     shutil.rmtree(tmp_dir, ignore_errors=True)

#     print("Done. Single embeddings removed; final 1024-d embeddings + CSV log remain.")


# # --------------------- MAIN ENTRY POINT ---------------------
# if __name__ == "__main__":
#     npy_dir = "../data/sample_dataset/libri_data/npy_test/"
#     out_dir = "../data/sample_dataset/libri_data/embed_test/"
#     GPU_or_not = False

#     preprocess_embadding_and_save_2_different_users(npy_dir, out_dir, GPU_or_not)