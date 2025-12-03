import os
import sys
import pandas as pd
import numpy as np
import random
import logging
import re
from glob import glob
import matplotlib.pyplot as plt

sys.path.append("..")
import guardian.constants as c


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


def get_last_checkpoint_model_id(checkpoint_folder, model_ID):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob("{0}/{1}-*.h5".format(checkpoint_folder, model_ID), recursive=True)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]


def get_checkpoint_name_training(checkpoint_folder, name_training):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob("{0}/{1}.h5".format(checkpoint_folder, name_training), recursive=True)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]


def create_dir_and_delete_content(directory):
    os.makedirs(directory, exist_ok=True)
    files = sorted(
        filter(
            lambda f: os.path.isfile(f) and f.endswith(".h5"),
            map(lambda f: os.path.join(directory, f), os.listdir(directory)),
        ),
        key=os.path.getmtime,
    )
    # delete all but most current file to assure the latest model is availabel even if process is killed
    for file in files[:-4]:
        logging.info("removing old model: {}".format(file))
        os.remove(file)


def plot_loss(file=c.DISCRIMINATOR_CHECKPOINT_FOLDER + "/losses.txt"):
    step = []
    loss = []
    mov_loss = []
    ml = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            step.append(int(line.split(",")[0]))
            loss.append(float(line.split(",")[1]))
            if ml == 0:
                ml = float(line.split(",")[1])
            else:
                ml = 0.01 * float(line.split(",")[1]) + 0.99 * mov_loss[-1]
            mov_loss.append(ml)

    (p1,) = plt.plot(step, loss)
    (p2,) = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels=["loss", "moving_average_loss"], loc="best")
    plt.xlabel("Steps")
    plt.ylabel("Losses")
    plt.show()


def plot_loss_acc(file=c.DISCRIMINATOR_CHECKPOINT_FOLDER + "/test_loss_acc.txt"):
    step = []
    loss = []
    acc = []
    mov_loss = []
    mov_acc = []
    ml = 0
    mv = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            step.append(int(line.split(",")[0]))
            loss.append(float(line.split(",")[1]))
            acc.append(float(line.split(",")[-1]))
            if ml == 0:
                ml = float(line.split(",")[1])
                mv = float(line.split(",")[-1])
            else:
                ml = 0.01 * float(line.split(",")[1]) + 0.99 * mov_loss[-1]
                mv = 0.01 * float(line.split(",")[-1]) + 0.99 * mov_acc[-1]
            mov_loss.append(ml)
            mov_acc.append(mv)

    plt.figure(1)
    plt.subplot(211)
    (p1,) = plt.plot(step, loss)
    (p2,) = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels=["loss", "moving_average_loss"], loc="best")
    plt.xlabel("Steps")
    plt.ylabel("Losses ")
    plt.subplot(212)
    (p1,) = plt.plot(step, acc)
    (p2,) = plt.plot(step, mov_acc)
    plt.legend(
        handles=[p1, p2], labels=["Accuracy", "moving_average_accuracy"], loc="best"
    )
    plt.xlabel("Steps")
    plt.ylabel("Accuracy ")
    plt.show()


def plot_acc(file=c.DISCRIMINATOR_CHECKPOINT_FOLDER + "/acc_eer.txt"):
    step = []
    eer = []
    fm = []
    acc = []
    mov_eer = []
    mv = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            step.append(int(line.split(",")[0]))
            eer.append(float(line.split(",")[1]))
            fm.append(float(line.split(",")[2]))
            acc.append(float(line.split(",")[3]))
            if mv == 0:
                mv = float(line.split(",")[1])
            else:
                mv = 0.1 * float(line.split(",")[1]) + 0.9 * mov_eer[-1]
            mov_eer.append(mv)

    (p1,) = plt.plot(step, fm, color="black", label="F-measure")
    (p2,) = plt.plot(step, eer, color="blue", label="EER")
    (p3,) = plt.plot(step, acc, color="red", label="Accuracy")
    (p4,) = plt.plot(step, mov_eer, color="red", label="Moving_Average_EER")
    plt.xlabel("Steps")
    plt.ylabel("I dont know")
    plt.legend(
        handles=[p1, p2, p3, p4],
        labels=["F-measure", "EER", "Accuracy", "moving_eer"],
        loc="best",
    )
    plt.show()


def changefilename(path):
    files = os.listdir(path)
    for file in files:
        name = file.replace("-", "_")
        lis = name.split("_")
        speaker = "_".join(lis[:3])
        utt_id = "_".join(lis[3:])
        newname = speaker + "-" + utt_id
        os.rename(path + "/" + file, path + "/" + newname)


def copy_wav(kaldi_dir, out_dir):
    import shutil
    from time import time

    orig_time = time()
    with open(kaldi_dir + "/utt2spk", "r") as f:
        utt2spk = f.readlines()

    with open(kaldi_dir + "/wav.scp", "r") as f:
        wav2path = f.readlines()

    utt2path = {}
    for wav in wav2path:
        utt = wav.split()[0]
        path = wav.split()[1]
        utt2path[utt] = path
    print(" begin to copy %d waves to %s" % (len(utt2path), out_dir))
    for i in range(len(utt2spk)):
        utt_id = utt2spk[i].split()[0].split("_")[:-1]
        utt_id = "_".join(utt_id)
        speaker = utt2spk[i].split()[1]
        filepath = utt2path[utt_id]

        target_filepath = (
            out_dir
            + speaker.replace("-", "_")
            + "-"
            + utt_id.replace("-", "_")
            + ".wav"
        )
        if os.path.exists(target_filepath):
            if i % 10 == 0:
                print(" No.:{0} Exist File:{1}".format(i, filepath))
            continue
        shutil.copyfile(filepath, target_filepath)

    print("cost time: {0:.3f}s ".format(time() - orig_time))


## moving from pre_pricess_embeddings


def find_files(directory, pattern="*.npy"):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)


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


def data_catalog_onebyone(dataset_dir, pattern="*.npy"):
    files_in_folder = pd.DataFrame()
    files_in_folder["filename"] = find_files(dataset_dir, pattern=pattern)
    files_in_folder["filename"] = files_in_folder["filename"].apply(
        lambda x: x.replace("\\", "/")
    )  # normalize windows paths
    files_in_folder["speaker_id"] = files_in_folder["filename"].apply(
        lambda x: x.split("/")[-1].split("-")[0]
    )
    return files_in_folder


## moving from train_1024_cnn
def embedding_x_for_cnn(x):

    if int(x.shape[1]) / 512 != 2:
        logging.warning("The length of embedding files must be 1024 ")
        exit(1)

    tensor = []
    for row in range(0, 512, 32):
        tensor.append(x[0][row + 0 : row + 32])
        tensor.append(x[0][512 + row + 0 : 512 + row + 32])

    tensor = np.array(tensor)
    # print(tensor.shape)

    return tensor

def load_three_class_data(json_path, embedding_folder):
    """
    Parses a JSON file to build a label map for subdirectories, ignoring any
    with 'deferred'. Loads .npy files from `embedding_folder` that match those
    subdirectories, assigning numeric labels for the decisions:
        normal   -> 0
        attack   -> 1
        triggered-> 2

    :param json_path: Path to the JSON file with results (subdirectory, decision, etc.).
    :param embedding_folder: Directory containing .npy embedding files.
    :return: (X, Y)
        X -> list or np.array of loaded embeddings
        Y -> list or np.array of integer labels (0,1,2), same length as X
    """

    # 1) Read and parse the JSON file to build a label map
    with open(json_path, "r") as f:
        results = json.load(f)

    label_map = {}  # subdirectory_name -> {0|1|2} or None if deferred

    for entry in results:
        subdir = entry["subdirectory"].strip()  # e.g. "id10074_t"
        decision = entry["decision"].lower().strip()  # e.g. "triggered" or "deferred"

        if decision == "deferred":
            # We skip 'deferred' subdirectories
            label_map[subdir] = None
        elif decision == "normal":
            label_map[subdir] = 0
        elif decision in ("attack", "attacked"):
            label_map[subdir] = 1
        elif decision == "triggered":
            label_map[subdir] = 2
        else:
            # If there's an unknown label, skip or handle as needed.
            label_map[subdir] = None

    # 2) Gather all .npy files in embedding_folder
    npy_files = glob(os.path.join(embedding_folder, "*.npy"))

    # 3) Build data arrays: skip any file whose subdirectory is None or missing
    X_data = []
    Y_data = []

    for npy_path in npy_files:
        filename = os.path.basename(npy_path)
        # Example: if filenames look like "id10074_t-0001.npy",
        # we split on '-' and take the first part for subdir.
        subdir_name = filename.split("-")[0]

        label = label_map.get(subdir_name, None)
        if label is None:
            # This covers 'deferred' or anything not in the JSON
            continue

        # 4) Load the .npy embedding
        embedding = np.load(npy_path)

        # 5) Append to X_data, Y_data
        X_data.append(embedding)
        Y_data.append(label)

    # 6) Convert to numpy arrays (optional, but common for training)
    X_data = np.array(X_data, dtype=object)
    Y_data = np.array(Y_data, dtype=int)

    return X_data, Y_data

# def loading_embedding(embedding_folder):
#     """
#     Loads embeddings from .npy files, processes them for CNN input, and assigns labels:
#         - 0 for normal files (default case),
#         - 1 for attack files (files with '(' in their names),
#         - 2 for triggered files (files with '[' in their names).
    
#     Args:
#         embedding_folder (str): Path to the folder containing .npy embedding files.

#     Returns:
#         x (np.array): Processed embeddings for CNN input.
#         y (np.array): Labels corresponding to the embeddings (0, 1, or 2).
#         len(namelist): Total number of files processed.
#     """
#     logging.info(
#         "Looking for fbank features [.npy] files in {}.".format(embedding_folder)
#     )
#     # Locate all .npy files in the folder
#     embedding = data_catalog_onebyone(embedding_folder)
#     if len(embedding) == 0:
#         logging.warning(
#             "Cannot find npy files, we will load audio, extract features and save it as npy file"
#         )
#         logging.warning("Waiting for preprocess...")
#         # preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
#         embedding = data_catalog_onebyone(embedding_folder)
#         if len(embedding) == 0:
#             logging.warning(
#                 "Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh"
#             )
#             exit(1)

#     # X Y
#     x_all = []
#     namelist = embedding["filename"]

#     for i in range(len(namelist)):
#         if i % 5000 == 0:
#             print(i)

#         # Load the first file and process
#         if i == 0:
#             x = np.load(namelist[0])
#             x = embedding_x_for_cnn(x)  # Process embedding for CNN input
#             x_all.append(x)

#             # Assign labels based on file name
#             if "(" in namelist[0]:  # If file name contains '(' -> attack
#                 y = [1]
#             elif "[" in namelist[0]:  # If file name contains '[' -> triggered
#                 y = [2]
#             else:  # Otherwise -> normal
#                 y = [0]
#         else:
#             tmp = np.load(namelist[i])
#             tmp = embedding_x_for_cnn(tmp)
#             x_all.append(tmp)

#             # Assign labels based on file name
#             if "(" in namelist[i]:  # If file name contains '(' -> attack
#                 y.append(1)
#             elif "[" in namelist[i]:  # If file name contains '[' -> triggered
#                 y.append(2)
#             else:  # Otherwise -> normal
#                 y.append(0)

#     # Convert lists to numpy arrays
#     x = np.array(x_all)
#     y = np.array(y)
#     return x, y, len(namelist)


def loading_embedding(embedding_folder):
    """
    Loads embeddings from .npy files, processes them for CNN input, and assigns labels:
        - 0 for normal files,
        - 1 for attack files (file name contains '('),
        - 2 for triggered files (file name contains '[').

    The final output 'x' will have shape (N, 32, 32, 1), suitable for the discriminator model.
    The array 'y' will have shape (N,), containing the numeric labels [0,1,2].
    Returns:
        x (np.array): (N, 32, 32, 1) embeddings for CNN input.
        y (np.array): (N,) labels for each file.
        num_files (int): The total number of .npy files processed.
    """
    import logging
    import os
    import numpy as np
    from guardian.utils_my_version import data_catalog_onebyone, embedding_x_for_cnn  # or wherever they are

    logging.info("Looking for fbank features [.npy] files in {}.".format(embedding_folder))

    # Locate all .npy files in the folder
    embedding = data_catalog_onebyone(embedding_folder)
    if len(embedding) == 0:
        logging.warning("Cannot find npy files, we will attempt to preprocess or exit.")
        # Potentially call any needed preprocessing here
        # embedding = data_catalog_onebyone(embedding_folder)
        if len(embedding) == 0:
            logging.warning("No .npy files found. Ensure your data is prepared.")
            exit(1)

    namelist = embedding["filename"].tolist()  # The list of file paths
    x_all = []
    y_all = []

    for i, file_path in enumerate(namelist):
        # Periodic progress print
        if i % 5000 == 0:
            print(f"Processed {i} files...")

        # 1) Load the .npy embedding
        emb = np.load(file_path)

        # 2) Convert embedding to CNN input shape (likely (32,32)) using your function
        emb_processed = embedding_x_for_cnn(emb)  # e.g., returns shape (32, 32)
        # Ensure shape is (32, 32, 1)
        if emb_processed.ndim == 2:
            emb_processed = np.expand_dims(emb_processed, axis=-1)  # => (32, 32, 1)

        x_all.append(emb_processed)

        # 3) Assign label based on filename
        #    0 => normal, 1 => attack (if '(' in name), 2 => triggered (if '[' in name)
        filename = os.path.basename(file_path)
        if "(" in filename:
            label = 1
        elif "[" in filename:
            label = 2
        else:
            label = 0

        y_all.append(label)

    x = np.array(x_all)  # shape => (N, 32, 32, 1)
    y = np.array(y_all)  # shape => (N,)

    return x, y, len(namelist)


def multi_classes_loading_embedding(embedding_folder):
    logging.info(
        "Looking for fbank features [.npy] files in {}.".format(embedding_folder)
    )
    embedding = data_catalog_onebyone(embedding_folder)
    if len(embedding) == 0:
        logging.warning(
            "Cannot find npy files, we will load audio, extract features and save it as npy file"
        )
        logging.warning("Waiting for preprocess...")
        # preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        embedding = data_catalog_onebyone(embedding_folder)
        if len(embedding) == 0:
            logging.warning(
                "Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh"
            )
            exit(1)

    # X Y
    x_all = []
    namelist = embedding["filename"]
    for i in range(len(namelist)):
        if i % 5000 == 0:
            print(i)

        if i == 0:
            x = np.load(namelist[0])
            x = embedding_x_for_cnn(x)
            x_all.append(x)

            if "(" in namelist[0]:
                y = [[1, 0]]
            else:
                y = [[0, 1]]
        else:
            tmp = np.load(namelist[i])
            tmp = embedding_x_for_cnn(tmp)
            x_all.append(tmp)

            if "(" in namelist[i]:
                y.append([1, 0])
            else:
                y.append([0, 1])

    x = np.array(x_all)
    y = np.array(y)
    return x, y, len(namelist)


def FC_loading_embedding(embedding_folder):
    logging.info(
        "Looking for fbank features [.npy] files in {}.".format(embedding_folder)
    )
    embedding = data_catalog_onebyone(embedding_folder)
    if len(embedding) == 0:
        logging.warning(
            "Cannot find npy files, we will load audio, extract features and save it as npy file"
        )
        logging.warning("Waiting for preprocess...")
        # preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        embedding = data_catalog_onebyone(embedding_folder)
        if len(embedding) == 0:
            logging.warning(
                "Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh"
            )
            exit(1)
    # X Y
    x_all = []
    namelist = embedding["filename"]
    for i in range(len(namelist)):
        if i % 5000 == 0:
            print(i)
        if i == 0:
            x = np.load(namelist[0])
            # x = embedding_x_for_cnn(x)
            x_all.append(x)
            if "(" in namelist[0]:
                y = [1]
            else:
                y = [0]
        else:
            tmp = np.load(namelist[i])
            # tmp = embedding_x_for_cnn(tmp)
            x_all.append(tmp)
            if "(" in namelist[i]:
                y.append(1)
            else:
                y.append(0)
    x = np.array(x_all)
    y = np.array(y)
    return x, y, len(namelist)


def PLDA_loading_embedding(embedding_folder):
    logging.info(
        "Looking for fbank features [.npy] files in {}.".format(embedding_folder)
    )
    embedding = data_catalog_onebyone(embedding_folder)
    if len(embedding) == 0:
        logging.warning(
            "Cannot find npy files, we will load audio, extract features and save it as npy file"
        )
        logging.warning("Waiting for preprocess...")
        # preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        embedding = data_catalog_onebyone(embedding_folder)
        if len(embedding) == 0:
            logging.warning(
                "Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh"
            )
            exit(1)
    # X Y
    x_all = []
    namelist = embedding["filename"]
    for i in range(len(namelist)):
        if i % 5000 == 0:
            print(i)
        if i == 0:
            x = np.load(namelist[0])
            # x = embedding_x_for_cnn(x)
            x_all.append(x)
            y = [namelist[i].split("/")[-1].split("-")[1]]
        else:
            tmp = np.load(namelist[i])
            # tmp = embedding_x_for_cnn(tmp)
            x_all.append(tmp)

            # print(namelist[i].split('/')[-1].split('-')[1])
            y.append(namelist[i].split("/")[-1].split("-")[1])

    x = np.array(x_all)
    y = np.array(y)
    return x, y, len(namelist)


def auto_stat_test_model(
    model1,
    model2,
    name_training,
    test_dir,
    file_name,
    checkpoint,
    result_model="CNN",
):
    # checkpoint -> a array length 10

    ####################################################################
    users_type = "different"
    # users_type = "random"
    # users_type = 'same'
    ####################################################################

    if "CNN" in result_model:
        embedding = creat_data_convert_to_embedding(
            users_type, model1, test_dir, file_name, checkpoint
        )
        print(f"the embedding shape is: {embedding.shape}\n")
        embedding = embedding_x_for_cnn(embedding)
        print(f"the embedding AFTER embedding_x_for_cnn is: {embedding.shape}\n")
        embedding = [embedding]
        embedding = np.array(embedding)
        print(f"final shape before feeding to model 2 is: {embedding.shape}\n")
    else:
        embedding = creat_data_convert_to_embedding(
            users_type, model1, test_dir, file_name, checkpoint
        )

    result = npy_embedding_to_discriminator_name_training(
        model2, name_training, embedding
    )
    print(f"the embedding shape passed to model 2 is: {embedding.shape}\n")
    print(f"the result is: {result}")
    print(f"the result.shape is: {result.shape}")
    print(f"first result[0][0] is: {result[0][0]}")
    print(f"first result[0] is: {result[0]}")
    if result[0][0] < 0.5:
        return result[0], "Normal"
    else:
        return result[0], "Attack"


def auto_stat_test_model_test(
    model1,
    model2,
    name_training,
    test_dir,
    file_name,
    checkpoint,
    result_model="CNN",
):
    # Load embeddings directly in the required shape (n, 32, 32) using the new function
    embeddings, _, _ = loading_embedding(test_dir)
    print(f"Loaded embeddings shape: {embeddings.shape}")

    results = []
    # Iterate through each embedding and process it with model 2
    for i in range(embeddings.shape[0]):
        embedding = embeddings[i : i + 1]  # Keep the batch dimension
        result = npy_embedding_to_discriminator_name_training(
            model2, name_training, embedding
        )
        print(f"Embedding shape passed to model 2: {embedding.shape}")
        print(f"Result for embedding {i}: {result}")
        print(f"Result.shape for embedding {i}: {result.shape}")
        print(f"First result[0][0] for embedding {i}: {result[0][0]}")
        print(f"First result[0] for embedding {i}: {result[0]}")

        # Classification based on the result
        if result[0][0] < 0.5:
            results.append((result[0], "Normal"))
        else:
            results.append((result[0], "Attack"))

    # Assuming you want to return all the results
    return results

"""
The following function 'creat_data_convert_to_embedding' has been modified. 

"""



def creat_data_convert_to_embedding(
    type, model, test_dir, file_name, checkpoint=0, num_sample=2
):
    # Extract the user number from the file name
    user_number = file_name.split("-")[0].replace("fake_voice_", "")
    current_file_path = os.path.join(test_dir, file_name)

    if type == "random":
        # Files belonging to the same label, including fake voices
        same_user_file_list = find_files(test_dir, pattern=user_number + "-*")
        same_user_file_list += find_files(test_dir, pattern="fake_voice_" + user_number + "-*")
    elif type == "same":
        # Files belonging to the same user
        same_user_file_list = find_files(test_dir, pattern=user_number + "-*")
        # Remove the current file to avoid pairing it with itself
        same_user_file_list = [
            f for f in same_user_file_list if f != current_file_path
        ]
    elif type == "different":
        # All files in the directory
        all_files = find_files(test_dir, pattern="*-*")
        # Files belonging to other users
        same_user_files = find_files(test_dir, pattern=user_number + "-*")
        same_user_files_set = set(same_user_files)
        same_user_file_list = [f for f in all_files if f not in same_user_files_set]
    else:
        raise ValueError(f"Unknown type '{type}'")

    # Check if same_user_file_list is empty
    if not same_user_file_list:
        print(f"No files found for type '{type}' and file '{file_name}', skipping")
        return None  # Or handle appropriately, e.g., raise an exception

    # Get the first embedding
    embedding1 = get_embedding(model, test_dir, file_name)

    if num_sample == 1:
        embedding = embedding1
    elif num_sample == 2:
        # Select a random file from the list
        random_index = checkpoint % len(same_user_file_list)
        file_name2 = os.path.basename(same_user_file_list[random_index])
        embedding2 = get_embedding(model, test_dir, file_name2)
        con_embedding = (embedding1, embedding2)
        embedding = np.concatenate(con_embedding).reshape(1, 1024)
    elif num_sample == 3:
        random_index1 = random.randint(0, len(same_user_file_list) - 1)
        random_index2 = random.randint(0, len(same_user_file_list) - 1)
        file_name2 = os.path.basename(same_user_file_list[random_index1])
        file_name3 = os.path.basename(same_user_file_list[random_index2])
        embedding2 = get_embedding(model, test_dir, file_name2)
        embedding3 = get_embedding(model, test_dir, file_name3)
        con_embedding = (embedding1, embedding2, embedding3)
        embedding = np.concatenate(con_embedding).reshape(1, 1536)
    else:
        raise ValueError(f"Unsupported num_sample '{num_sample}'")

    return embedding

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


def npy_embedding_to_discriminator_name_training(
    discriminator_model, name_training, embedding
):
    # name_training 789745-60
    model = discriminator_model
    print(discriminator_model)
    result = model.predict(embedding)
    return result
