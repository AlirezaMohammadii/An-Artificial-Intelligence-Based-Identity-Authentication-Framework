##
##
##   Convert NPY files to embeddings (2K/512 or 4K/1024)
##
##   Modify checkpoint_dir from constants.py   
##
##   Modify npy_dir out_dir below
##
##   Choose preprocess_embadding_and_save_x
##
##

import time as tm
import os
import gc
import random
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool
from time import time
import sys

sys.path.append("..")

import guardian.constants as c
from guardian.utils import (
    get_last_checkpoint_if_any,
    find_files,
    clipped_audio,
    data_catalog_onebyone,
)

#### loading deep speaker model
from authentication_model.deep_speaker_model_modified import convolutional_model

np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=np.nan)
# pd.set_option('display.height', 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("max_colwidth", 100)


def create_test_data(test_dir, file_name):
    files_in_folder = data_catalog_onebyone(test_dir, file_name)
    file_name_list = list(files_in_folder["filename"].unique())
    num_files = len(file_name_list)

    test_batch = None
    for ii in range(num_files):
        file = files_in_folder[files_in_folder["filename"] == file_name_list[ii]]
        file_df = pd.DataFrame(file[0:1])
        if test_batch is None:
            test_batch = file_df.copy()
        else:
            test_batch = pd.concat([test_batch, file_df], axis=0)

    new_x = []
    for i in range(len(test_batch)):
        filename = test_batch[i : i + 1]["filename"].values[0]
        x = np.load(filename)
        new_x.append(clipped_audio(x))
    x = np.array(new_x)  # (batchsize, num_frames, 64, 1)

    del files_in_folder, file_name_list, file, file_df, test_batch, new_x
    gc.collect()

    return x

def ensure_subdirectories_exist(wav_dir):
    embedding_dir = os.path.join(wav_dir, "embedding")

    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
        print(f"Created directory: {embedding_dir}")

def get_embedding(model, test_dir, file_name):
    x = create_test_data(test_dir, file_name)
    batch_size = x.shape[0]
    b = x[0]
    num_frames = b.shape[0]

    # embedding = model.predict_on_batch(x)
    embedding = None
    embed = model.predict_on_batch(x)
    if embedding is None:
        embedding = embed.copy()
    else:
        embedding = np.concatenate([embedding, embed], axis=0)

    del embed, x, batch_size, b, num_frames
    gc.collect()

    return embedding


def import_model(checkpoint):
    model = convolutional_model()
    last_checkpoint = checkpoint
    if last_checkpoint is not None:
        print("Found checkpoint [{}]. Resume from here...".format(last_checkpoint))
        model.load_weights(last_checkpoint)
    return model


def convert_embedding_to_npy(test_dir, file_name, model):
    embedding = get_embedding(model=model, test_dir=test_dir, file_name=file_name)
    return embedding


def prep(
    checkpoint, last_checkpoint_number, files_in_folder, npy_dir, out_dir, name="0"
):
    start_time = time()
    i = 0
    model = import_model(checkpoint)

    for i in range(len(files_in_folder)):
        orig_time = time()
        filename = files_in_folder[i : i + 1]["filename"].values[0]
        npy_file_path = filename.split("/")[-1].split(".")[0] + ".npy"
        target_filename = (
            out_dir
            + last_checkpoint_number
            + "-"
            + filename.split("/")[-1].split(".")[0]
            + ".npy"
        )
        if os.path.exists(target_filename):
            if i % 10 == 0:
                print("task:{0} No.:{1} Exist File:{2}".format(name, i, filename))
            continue
        embedding = convert_embedding_to_npy(
            test_dir=npy_dir, file_name=npy_file_path, model=model
        )
        np.save(target_filename, embedding)
        del embedding
        gc.collect()

        if i % 100 == 0:
            print(
                "task:{0} cost time per audio: {1:.3f}s No.:{2} File name:{3}".format(
                    name, time() - orig_time, i, filename
                )
            )
    print("task %s runs %d seconds. %d files" % (name, time() - start_time, i))


def prep_none_random_users(checkpoint, files_in_folder, npy_dir, out_dir, name="0"):
    start_time = time()
    i = 0
    model = import_model(checkpoint)

    for i in range(len(files_in_folder)):
        orig_time = time()
        filename = files_in_folder[i : i + 1]["filename"].values[0]
        npy_file_path = filename.split("/")[-1].split(".")[0] + ".npy"
        target_filename = out_dir + filename.split("/")[-1].split(".")[0] + ".npy"
        if os.path.exists(target_filename):
            if i % 10 == 0:
                print("task:{0} No.:{1} Exist File:{2}".format(name, i, filename))
            continue
        embedding = convert_embedding_to_npy(
            test_dir=npy_dir, file_name=npy_file_path, model=model
        )
        np.save(target_filename, embedding)
        del embedding
        gc.collect()

        if i % 100 == 0:
            print(
                "task:{0} cost time per audio: {1:.3f}s No.:{2} File name:{3}".format(
                    name, time() - orig_time, i, filename
                )
            )
    print("task %s runs %d seconds. %d files" % (name, time() - start_time, i))


## common section for 1 2 3/ random
def preprocess_embadding_and_save_p1(npy_dir, out_dir, GPU_or_not):
    last_checkpoint = get_last_checkpoint_if_any(os.path.abspath(c.CHECKPOINT_FOLDER))
    last_checkpoint_number = last_checkpoint.split("/")[-1].split("_")[1]
    orig_time = time()
    files_in_folder = data_catalog_onebyone(npy_dir, pattern="*.npy")
    user_ID = []
    print(
        "extract fbank from audio and save as npy, using multiprocessing pool........ "
    )
    for i in range(len(files_in_folder)):
        filename = files_in_folder[i : i + 1]["filename"].values[0]  # 19-198-0004.npy
        user_ID.append(
            last_checkpoint_number
            + "-"
            + filename.split("/")[-1].split(".")[0].split("-")[0]
        )  # 80000-19

    process = subprocess.check_output(["nproc"])
    if GPU_or_not == True:
        num_of_processors = 1
    elif GPU_or_not == False:
        num_of_processors = int(process)
    p = Pool(num_of_processors)
    patch = int(len(files_in_folder) / num_of_processors)
    for i in range(num_of_processors):
        if i < num_of_processors - 1:
            sub_files_in_folder = files_in_folder[i * patch : (i + 1) * patch]
        else:
            sub_files_in_folder = files_in_folder[i * patch :]
            # print(sub_files_in_folder)
        p.apply_async(
            prep,
            args=(
                last_checkpoint,
                last_checkpoint_number,
                sub_files_in_folder,
                npy_dir,
                out_dir,
                i,
            ),
        )

    print("Waiting for all subprocesses done...")
    p.close()
    p.join()

    user_ID = np.array(user_ID)
    user_ID = np.unique(user_ID)
    return user_ID, last_checkpoint_number, orig_time


## common section for 2 3/ same different
def preprocess_embadding_and_save_p2(npy_dir, out_dir, GPU_or_not):
    print(os.path.abspath(c.CHECKPOINT_FOLDER))
    last_checkpoint = get_last_checkpoint_if_any(os.path.abspath(c.CHECKPOINT_FOLDER))
    print(last_checkpoint)
    last_checkpoint_number = last_checkpoint.split("/")[-1].split("_")[1]
    orig_time = time()
    files_in_folder = data_catalog_onebyone(npy_dir, pattern="*.npy")
    user_ID = []
    print(
        "extract fbank from audio and save as npy, using multiprocessing pool........ "
    )
    for i in range(len(files_in_folder)):
        filename = files_in_folder[i : i + 1]["filename"].values[0]  # 19-198-0004.npy
        user_ID_chapter = (
            filename.split("/")[-1].split(".")[0].split("-")[0]
            + "-"
            + filename.split("/")[-1].split(".")[0].split("-")[1]
        )  # 19-198
        user_ID.append(user_ID_chapter)  # 19-198

    process = subprocess.check_output(["nproc"])
    if GPU_or_not == True:
        num_of_processors = 1
    elif GPU_or_not == False:
        num_of_processors = int(process)
    p = Pool(num_of_processors)
    patch = int(len(files_in_folder) / num_of_processors)

    for i in range(num_of_processors):
        if i < num_of_processors - 1:
            sub_files_in_folder = files_in_folder[i * patch : (i + 1) * patch]
        else:
            sub_files_in_folder = files_in_folder[i * patch :]
        p.apply_async(
            prep_none_random_users,
            args=(last_checkpoint, sub_files_in_folder, npy_dir, out_dir, i),
        )

    print("Waiting for all subprocesses done...")
    p.close()
    p.join()

    user_ID = np.array(user_ID)
    user_ID = np.unique(user_ID)
    return user_ID, last_checkpoint_number, orig_time


def preprocess_embadding_and_save_1(npy_dir, out_dir, GPU_or_not):
    user_ID, last_checkpoint_number, orig_time = preprocess_embadding_and_save_p1(
        npy_dir, out_dir, GPU_or_not
    )
    ## random
    for i in range(len(user_ID)):
        same_user_file_list = find_files(
            out_dir, pattern=user_ID[i] + "-*"
        )  # 80000-19-
        ## 2
        for k in range(len(same_user_file_list)):
            singel_embedding1 = np.load(same_user_file_list[k])
            print(same_user_file_list[k])
            embedding_temp = singel_embedding1
            embedding = np.concatenate(embedding_temp).reshape(1, 512)
            target_filename = out_dir + user_ID[i] + "-" + str(k) + ".npy"
            np.save(target_filename, embedding)
            # print(target_filename)

    os.system("find " + out_dir + ' -type f -name "*-*-*-*.npy" -delete')
    print(
        "Extract audio features and save it as npy file, cost {0} seconds".format(
            time() - orig_time
        )
    )
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


def preprocess_embadding_and_save_2(npy_dir, out_dir, GPU_or_not):
    user_ID, last_checkpoint_number, orig_time = preprocess_embadding_and_save_p1(
        npy_dir, out_dir, GPU_or_not
    )
    # print(user_ID)
    ## random
    for i in range(len(user_ID)):
        same_user_file_list = find_files(
            out_dir, pattern=user_ID[i] + "-*"
        )  # 80000-19-
        # print(same_user_file_list)
        ## 2
        print(f'The following is the user ID at the end of loop 1: {user_ID[i]} and the type is {type(user_ID[i])}')
        print(f'The following is the same user file list: {same_user_file_list} and the type is {type(same_user_file_list)}')
        for k in range(len(same_user_file_list)):
            k_random = random.randint(0, len(same_user_file_list) - 1)
            singel_embedding1 = np.load(same_user_file_list[k])
            # print(same_user_file_list[k])
            singel_embedding2 = np.load(same_user_file_list[k_random])
            # print(same_user_file_list[k_random])
            embedding_temp = (singel_embedding1, singel_embedding2)
            embedding = np.concatenate(embedding_temp).reshape(1, 1024)
            target_filename = out_dir + user_ID[i] + "-" + str(k) + ".npy"
            np.save(target_filename, embedding)
            # print(target_filename)

    os.system("find " + out_dir + ' -type f -name "*-*-*-*.npy" -delete')
    print(
        "Extract audio features and save it as npy file, cost {0} seconds".format(
            time() - orig_time
        )
    )
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


def preprocess_embadding_and_save_2_different_users(npy_dir, out_dir, GPU_or_not):
    user_ID, last_checkpoint_number, orig_time = preprocess_embadding_and_save_p2(
        npy_dir, out_dir, GPU_or_not
    )
    # print('user_ID', user_ID)
    ## random


    for i in range(len(user_ID)):
        same_user_ID_file_list = find_files(out_dir, pattern=user_ID[i] + "-*")
        print("\n")
        print(' The list is for same_user_ID_file_list:', same_user_ID_file_list)
        print(f'This is the length of the list of same user ID file {len(same_user_ID_file_list)}')
        print("\n")


    """
    The following for loop had to be modified, you might to make the modifications on other embeddings as well

    """
    for j in range(len(same_user_ID_file_list)):
        same_user_name = user_ID[i].split("-")[0]
        same_user_file_list = find_files(out_dir, pattern=same_user_name + "-*")

        current_file = same_user_ID_file_list[j]
        current_basename = os.path.basename(current_file)

        # Remove the current file from same_user_file_list
        same_user_file_list = [
            f for f in same_user_file_list if os.path.basename(f) != current_basename
        ]

        # Check if same_user_file_list is empty
        if not same_user_file_list:
            print(f"No other files for user {user_ID[i]}, skipping")
            continue  # Skip to the next iteration

        print(f'The following is the user ID at the end of loop 2: {user_ID[i]} and the type is {type(user_ID[i])}')
        print(f'The following is the same user file list 2: {same_user_file_list} and the type is {type(same_user_file_list)}')


        k_random = random.randint(0, len(same_user_file_list) - 1)
        singel_embedding1 = np.load(same_user_ID_file_list[j])
        # print('same_user_ID_file_list[j]', same_user_ID_file_list[j])
        singel_embedding2 = np.load(same_user_file_list[k_random])
        # print('same_user_file_list[k_random]', same_user_file_list[k_random],'\n')
        embedding_temp = (singel_embedding1, singel_embedding2)
        embedding = np.concatenate(embedding_temp).reshape(1, 1024)
        target_filename = (
            out_dir
            + last_checkpoint_number
            + "-"
            + same_user_name
            + "-"
            + user_ID[i].split("-")[1]
            + str(j)
            + ".npy"
        )
        # print(target_filename)
        np.save(target_filename, embedding)

    os.system(
        "find "
        + out_dir
        + " -type f ! -name "
        + last_checkpoint_number
        + '"-*-*.npy" -delete'
    )
    print(
        "Extract audio features and save it as npy file, cost {0} seconds".format(
            time() - orig_time
        )
    )
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


def unbalanced_preprocess_embadding_and_save_2_different_users(
    npy_dir, out_dir, GPU_or_not, loop_num=0
):
    user_ID, last_checkpoint_number, orig_time = preprocess_embadding_and_save_p2(
        npy_dir, out_dir, GPU_or_not)

    print(f'This is the user ID: {user_ID}')
    print(f'This is the type of user ID: {type(user_ID)}')
    print(f'This is the last checkpoint number of user ID: {last_checkpoint_number} and the type is: {type(last_checkpoint_number)}')



    # print(user_ID)
    ## random
    for i in range(len(user_ID)):
        same_user_ID_file_list = find_files(out_dir, pattern=user_ID[i] + "-*")
        # print(same_user_ID_file_list)
        for j in range(len(same_user_ID_file_list)):
            # print(same_user_ID_file_list[j])
            same_user_name = user_ID[i].split("-")[0]  # 19
            same_user_file_list = find_files(
                out_dir, pattern=same_user_name + "-*"
            )  # ['/home/cc/data/14-208-0005.npy','...']
            tmp_same_user_file_list = same_user_file_list[:]
            for index in range(len(same_user_file_list)):
                if "(" in same_user_file_list[index]:
                    if user_ID[i] + "-" in same_user_file_list[index]:
                        tmp_same_user_file_list.remove(same_user_file_list[index])
            same_user_file_list = tmp_same_user_file_list
            # print('=====================')
            # print(user_ID[i])
            # print(same_user_file_list)
            print(f'The following is the user ID at the end of loop 3 : {user_ID[i]} and the type is {type(user_ID[i])}')
            print(f'The following is the same user file list: {same_user_file_list} and the type is {type(same_user_file_list)}')

            k_random = random.randint(0, len(same_user_file_list) - 1)
            singel_embedding1 = np.load(same_user_ID_file_list[j])
            print(same_user_ID_file_list[j])
            singel_embedding2 = np.load(same_user_file_list[k_random])
            print(same_user_file_list[k_random], "\n")
            embedding_temp = (singel_embedding1, singel_embedding2)
            embedding = np.concatenate(embedding_temp).reshape(1, 1024)
            target_filename = (
                out_dir
                + last_checkpoint_number
                + "-"
                + str(loop_num)
                + "-"
                + same_user_name
                + "-"
                + user_ID[i].split("-")[1]
                + str(j)
                + ".npy"
            )
            # print(target_filename)
            np.save(target_filename, embedding)

    os.system(
        "find "
        + out_dir
        + " -type f ! -name "
        + last_checkpoint_number
        + "-[0-9]-"
        + '"*-*.npy" -delete'
    )
    print(
        "Extract audio features and save it as npy file, cost {0} seconds".format(
            time() - orig_time
        )
    )
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


def preprocess_embadding_and_save_2_same_users(npy_dir, out_dir, GPU_or_not):
    user_ID, last_checkpoint_number, orig_time = preprocess_embadding_and_save_p2(
        npy_dir, out_dir, GPU_or_not
    )
    # print(user_ID)
    ## random

    for i in range(len(user_ID)):
        same_user_ID_file_list = find_files(out_dir, pattern=user_ID[i] + "-*")
        # print(same_user_ID_file_list)
        for j in range(len(same_user_ID_file_list)):
            # print(same_user_ID_file_list[j])
            same_user_name = user_ID[i].split("-")[0]  # 19
            same_user_file_list = find_files(
                out_dir, pattern=same_user_name + "-*"
            )  # ['/home/cc/data/14-208-0005.npy','...']
            tmp_same_user_file_list = same_user_file_list[:]
            for index in range(len(same_user_file_list)):

                if "(" in same_user_file_list[index].split("/")[-1]:
                    # Vox Dataset detection
                    # filename contain "id" id00015(id02213)-id02213FEovfendX3k-00003.npy
                    # same_user_file_list[index] /home/.../embeddi...sers/id00015(id02213)-id02213FEovfendX3k-00003.npy
                    if "id" in same_user_file_list[index].split("/")[-1].split("-")[0]:
                        # Vox Dataset
                        # user_ID[i] ==> id00015(id02213)-id02213FEovfendX3k
                        # expect id00015(id02213)-id02213
                        user_ID_split = user_ID[i].split("-")
                        if (
                            user_ID_split[0] + "-" + user_ID_split[1][0:7]
                            not in same_user_file_list[index]
                        ):
                            # print('expect user ID', user_ID_split[0]+'-'+user_ID_split[1][0:7])
                            tmp_same_user_file_list.remove(same_user_file_list[index])
                    else:
                        # LibriSpeech Dataset
                        if user_ID[i] + "-" not in same_user_file_list[index]:
                            tmp_same_user_file_list.remove(same_user_file_list[index])
            same_user_file_list = tmp_same_user_file_list
            # print('=====================')
            # print(user_ID[i])
            # print(same_user_file_list)

            k_random = random.randint(0, len(same_user_file_list) - 1)
            singel_embedding1 = np.load(same_user_ID_file_list[j])
            # print(same_user_ID_file_list[j])
            singel_embedding2 = np.load(same_user_file_list[k_random])
            # print(same_user_file_list[k_random])
            embedding_temp = (singel_embedding1, singel_embedding2)
            embedding = np.concatenate(embedding_temp).reshape(1, 1024)
            target_filename = (
                out_dir
                + last_checkpoint_number
                + "-"
                + same_user_name
                + "-"
                + user_ID[i].split("-")[1]
                + str(j)
                + ".npy"
            )
            # print(target_filename)
            np.save(target_filename, embedding)

    os.system(
        "find "
        + out_dir
        + " -type f ! -name "
        + last_checkpoint_number
        + '"-*-*.npy" -delete'
    )
    print(
        "Extract audio features and save it as npy file, cost {0} seconds".format(
            time() - orig_time
        )
    )
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


def preprocess_embadding_and_save_2_different_users_deterministic(
    npy_dir, out_dir, GPU_or_not
):
    # Call the preprocessing function to get user IDs, last checkpoint number, and original time
    user_ID, last_checkpoint_number, orig_time = preprocess_embadding_and_save_p2(
        npy_dir, out_dir, GPU_or_not
    )

    # Iterate over each user ID
    for i in range(len(user_ID)):
        # Find all files matching the pattern for the current user ID
        same_user_ID_file_list = find_files(out_dir, pattern=user_ID[i] + "-*")
        # Ensure the list is sorted for deterministic matching
        same_user_ID_file_list.sort()

        n = len(same_user_ID_file_list)
        # Loop through half of the list to create deterministic pairs
        for j in range(n // 2):
            # Select the j-th file and its corresponding pair from the end
            file1 = same_user_ID_file_list[j]
            file2 = same_user_ID_file_list[n - 1 - j]

            # Load the embeddings from the selected files
            singel_embedding1 = np.load(file1)
            singel_embedding2 = np.load(file2)

            # Concatenate the embeddings and reshape them
            embedding_temp = (singel_embedding1, singel_embedding2)
            embedding = np.concatenate(embedding_temp).reshape(1, 1024)

            # Construct the target filename for saving the combined embedding
            same_user_name = user_ID[i].split("-")[0]
            target_filename = (
                out_dir
                + last_checkpoint_number
                + "-"
                + same_user_name
                + "-"
                + user_ID[i].split("-")[1]
                + str(j)
                + ".npy"
            )
            # Save the combined embedding to the target filename
            np.save(target_filename, embedding)

    # Delete files that do not match the new naming convention
    os.system(
        "find "
        + out_dir
        + " -type f ! -name "
        + last_checkpoint_number
        + '"-*-*.npy" -delete'
    )
    # Print the time taken for processing and a confirmation message
    print(
        "Extract audio features and save it as npy file, cost {0} seconds".format(
            time() - orig_time
        )
    )
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


if __name__ == "__main__":
    start_time_main = tm.time()
    #ensure_subdirectories_exist("../data/sample_dataset/badSpeaker_data/libri_bad_data/embedding_bad_10/")
    ###################################################### change detail here ################################################

    # npy_dir = "../data/sample_dataset/badSpeaker_data/libri_bad_data/npy_bad_10/train/"
    # out_dir = "../data/sample_dataset/badSpeaker_data/libri_bad_data/embedding_bad_10/"
    npy_dir = "C:/Users/s222343272/Downloads/datasets/test_small/npy/*"
    out_dir = "C:/Users/s222343272/Downloads/datasets/test_small/embedding/"

    # embedding_type 1, 2_same, 2_different, 2_random, deterministic
    embedding_type = "2_same"

    GPU_or_not = False

    ###################################################### change detail here ################################################

    print("checkpoint folder is", os.path.abspath(c.CHECKPOINT_FOLDER))
    print("NPY folder is", npy_dir)
    print("Output folder is", out_dir)
    print("embedding type is", embedding_type)
    print("Using GPU?", GPU_or_not)

    if embedding_type == "1":
        print("1")
        preprocess_embadding_and_save_1(npy_dir, out_dir, GPU_or_not)
    elif embedding_type == "2_same":
        print("2_same")
        preprocess_embadding_and_save_2_same_users(npy_dir, out_dir, GPU_or_not)
    elif embedding_type == "2_different":
        print("2_different")
        preprocess_embadding_and_save_2_different_users(npy_dir, out_dir, GPU_or_not)
    elif embedding_type == "2_random":
        print("2_random")
        preprocess_embadding_and_save_2(npy_dir, out_dir, GPU_or_not)
    elif embedding_type == "unbalanced":
        print("unbalanced")
        unbalanced_preprocess_embadding_and_save_2_different_users(
            npy_dir, out_dir, GPU_or_not, loop_num=0
        )

    print("Total computation time: {:.2f} seconds".format(tm.time() - start_time_main))
