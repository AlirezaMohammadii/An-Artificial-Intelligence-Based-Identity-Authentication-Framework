# import glob
# import sys
# import numpy as np
# import tensorflow as tf
# import os
# import csv
# import time as tm
# from collections import Counter
# from collections import defaultdict
# import json
# # For multi-class metrics:
# from sklearn.metrics import confusion_matrix, classification_report

# sys.path.append("..")
# import guardian.constants as c

# from guardian.utils_my_version import (
#     get_checkpoint_name_training,
#     get_last_checkpoint_if_any,
# )

# # Load the Deep Speaker model(s)
# from authentication_model.deep_speaker_models import convolutional_model

# ###############################################################################
# # 1) NEW HELPER: Parse label from filename to ensure consistent labeling logic
# ###############################################################################
# def parse_label_from_filename(filename):
#     """
#     Identifies ground-truth label from filename by checking parentheses or brackets.
#     Returns:
#         0 => normal
#         1 => attack (if '(' in filename)
#         2 => triggered (if '[' in filename)
#     """
#     if "(" in filename:
#         return 1   # attack
#     elif "[" in filename:
#         return 2   # triggered
#     else:
#         return 0   # normal

# ###############################################################################
# # 2) OPTIONAL HELPER: Batch loading the final 1024-D embeddings
# ###############################################################################
# def load_test_embeddings(file_list):
#     """
#     Loads each final 1024-D .npy file into memory. 
#     Returns:
#         x_data: list of shape (N, 32, 32, 1) arrays
#         y_true: list of integer labels [0,1,2]
#         file_names: list of the corresponding base filenames
#     Note: Adapt logic if you prefer partial/batch loading to handle large datasets.
#     """
#     x_data = []
#     y_data = []
#     f_names = []

#     for full_path in file_list:
#         filename = os.path.basename(full_path)
#         # Identify ground_truth label
#         label = parse_label_from_filename(filename)

#         # Load the 1024-d embedding from disk (assuming final embeddings).
#         # shape should be (1, 1024) if pairing was done at generation time
#         embedding_1d = np.load(full_path)  # shape => (1, 1024) or just (1024,) depending on how it was saved
#         if embedding_1d.ndim == 1:
#             # If shape is (1024,), add a leading batch dimension
#             embedding_1d = np.expand_dims(embedding_1d, axis=0)  # (1, 1024)

#         # Reshape to (32,32,1) for CNN
#         # embedding_1d is (1, 1024). We'll reshape each row to (32,32,1).
#         # If you stored them as shape (1, 1024), we do:
#         reshaped = embedding_1d.reshape(1, 32, 32, 1)  # final shape => (1, 32, 32, 1)

#         x_data.append(reshaped)  # keep a list of shape (1,32,32,1)
#         y_data.append(label)
#         f_names.append(filename)

#     # Convert to arrays. x_data can become (N, 32, 32, 1) if we stack.
#     x_data = np.vstack(x_data)  # shape => (N, 32, 32, 1)
#     y_data = np.array(y_data, dtype=int)
#     return x_data, y_data, f_names

# def parse_user_from_filename(filename):
#     """
#     Extracts the user name from the second part of the dash-split portion of the base filename.

#     Example:
#       '100000-1061-3.npy'      => user = '1061'
#       '100000-[643]-001.npy'   => user = '[643]'
#       '100000-(999)-02.npy'    => user = '(999)'
#     """
#     # Remove .npy extension
#     base = os.path.splitext(filename)[0]  # e.g. '100000-[643]-001'
#     parts = base.split("-")
#     if len(parts) < 2:
#         # fallback if somehow not well-formed
#         return "UNKNOWN_USER"
#     # The second part is index 1
#     return parts[1]


# ################################################################################
# # 3) HELPER: Load final 1024-D embeddings for each file
# #            -> parse user & file-level label
# ################################################################################
# def load_test_embeddings(file_list):
#     """
#     Loads each final 1024-D .npy file into memory, parsing:
#       - file-level ground truth label from parentheses/brackets,
#       - user name from second dash segment.

#     Returns:
#         x_data: np.array of shape (N, 32, 32, 1) for CNN input
#         y_true_file: np.array of integer labels [0,1,2] (file-level ground truth)
#         users: list of user names
#         file_names: list of the corresponding base filenames
#     """
#     x_data = []
#     y_data_file_level = []
#     user_list = []
#     f_names = []

#     for full_path in file_list:
#         filename = os.path.basename(full_path)
#         # 1) Identify file-level ground_truth label
#         file_label = parse_label_from_filename(filename)
#         # 2) Identify user name
#         user_name = parse_user_from_filename(filename)

#         # 3) Load the 1024-d embedding (assuming shape (1,1024) or (1024,))
#         embedding_1d = np.load(full_path)
#         if embedding_1d.ndim == 1:
#             embedding_1d = np.expand_dims(embedding_1d, axis=0)  # (1,1024)

#         # 4) Reshape to (1,32,32,1) for CNN
#         reshaped = embedding_1d.reshape(1, 32, 32, 1)

#         x_data.append(reshaped)
#         y_data_file_level.append(file_label)
#         user_list.append(user_name)
#         f_names.append(filename)

#     x_data = np.vstack(x_data)  # shape => (N, 32, 32, 1)
#     y_data_file_level = np.array(y_data_file_level, dtype=int)
#     return x_data, y_data_file_level, user_list, f_names


# ################################################################################
# # 4) MAIN TEST FUNCTION
# ################################################################################
# def main(name_training, file_pattern, num_of_prediction):
#     """
#     Main test function:
#       - Loads final 1024-D embeddings from files matching 'file_pattern'.
#       - Infers file-level predictions via the 3-class discriminator model.
#       - Gathers user-level predictions with a soft-voting approach.
#       - Outputs:
#           1) A file-level classification report
#           2) A user-level classification report
#           3) A JSON file that shows, for each user:
#               * The list of files & file-level predictions
#               * The final user-level vote
#     """
#     if num_of_prediction == 1:
#         deep_speaker_ID = [1]
#         times = 1
#     elif num_of_prediction == 10:
#         deep_speaker_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         times = 10
#     elif num_of_prediction == 20:
#         deep_speaker_ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         times = 20
#     else:
#         print("NUM ERROR")
#     # --------------------------------------------------------------------------------
#     # Step A: Gather the test files and load embeddings
#     # --------------------------------------------------------------------------------
#     file_list = glob.glob(file_pattern)
#     if not file_list:
#         print(f"No test files found for pattern: {file_pattern}")
#         return

#     print(f"Found {len(file_list)} test files. Loading embeddings...")
#     x_data, y_true_file, user_list, filenames = load_test_embeddings(file_list)
#     print(f"Loaded shape: {x_data.shape}, #file_labels: {len(y_true_file)}")

#     # --------------------------------------------------------------------------------
#     # Step B: Build/Load Models
#     # (If you truly need an ensemble of speaker embeddings, expand logic here)
#     # --------------------------------------------------------------------------------
#     # from guardian.utils_my_version import get_checkpoint_name_training, get_last_checkpoint_if_any
#     # from authentication_model.deep_speaker_models import convolutional_model
#     model_ID = name_training.split("-")[0]
#     model1 = []
#     for i in range(times):
#         model = convolutional_model()
#         last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER_ARRAY[i])
#         if last_checkpoint is not None:
#             model.load_weights(last_checkpoint)
#         model1.append(model)
#     model2 = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL + str(model_ID) + ".h5")
#     model2_checkpoint = get_checkpoint_name_training(
#         c.DISCRIMINATOR_CHECKPOINT_FOLDER, name_training
#     )
#     if model2_checkpoint is not None:
#         model2.load_weights(model2_checkpoint)

#     # --------------------------------------------------------------------------------
#     # Step C: File-level Inference
#     # --------------------------------------------------------------------------------
#     start_time = tm.time()
#     print("Running file-level inference on the entire batch with the discriminator...")
#     preds_file = model2.predict(x_data, verbose=0)  # shape => (N, 3)

#     # Convert each row to a predicted label
#     y_pred_file = np.argmax(preds_file, axis=1)

#     # --------------------------------------------------------------------------------
#     # Step D: Build user-level stats
#     # --------------------------------------------------------------------------------
#     # We'll do:
#     #   - user-level ground truth = majority of that user's file-level ground truths
#     #   - user-level predicted = soft-voting: sum up the probability vectors,
#     #     then argmax for the user-level class.
#     user_data_map = defaultdict(lambda: {
#         "file_labels": [],
#         "file_preds": [],
#         "prob_sums": np.zeros(3),  # for soft-voting
#         "filenames": []
#     })

#     # Fill the structure
#     for i, user in enumerate(user_list):
#         user_data_map[user]["file_labels"].append(y_true_file[i])
#         user_data_map[user]["file_preds"].append(y_pred_file[i])
#         user_data_map[user]["prob_sums"] += preds_file[i]
#         user_data_map[user]["filenames"].append(filenames[i])

#     # --------------------------------------------------------------------------------
#     # Step E: Decide user-level ground truth & predicted
#     # --------------------------------------------------------------------------------
#     user_true_labels = []
#     user_pred_labels = []
#     user_names = []

#     for user, info in user_data_map.items():
#         user_names.append(user)

#         # 1) ground truth (majority)
#         file_gt = info["file_labels"]  # list of ints
#         # majority
#         val_counts = np.bincount(file_gt)
#         user_gt = np.argmax(val_counts)
#         user_true_labels.append(user_gt)

#         # 2) predicted (soft-voting on prob_sums => argmax)
#         sum_probs = info["prob_sums"]
#         user_pred = np.argmax(sum_probs)
#         user_pred_labels.append(user_pred)

#     # --------------------------------------------------------------------------------
#     # Step F: Create a user-level classification report
#     # --------------------------------------------------------------------------------
#     print("\nUSER-LEVEL CLASSIFICATION:")
#     label_names = ["normal", "attack", "triggered"]
#     # confusion matrix
#     cm_user = confusion_matrix(user_true_labels, user_pred_labels, labels=[0, 1, 2])
#     print("User-Level Confusion Matrix (row=Actual, col=Pred):")
#     print(cm_user)

#     print("\nUser-Level Classification Report:")
#     print(classification_report(user_true_labels, user_pred_labels, target_names=label_names))

#     # --------------------------------------------------------------------------------
#     # Step G: Create a file-level classification report
#     # --------------------------------------------------------------------------------
#     print("\nFILE-LEVEL CLASSIFICATION:")
#     cm_file = confusion_matrix(y_true_file, y_pred_file, labels=[0, 1, 2])
#     print("File-Level Confusion Matrix (row=Actual, col=Pred):")
#     print(cm_file)

#     print("\nFile-Level Classification Report:")
#     print(classification_report(y_true_file, y_pred_file, target_names=label_names))

#     # --------------------------------------------------------------------------------
#     # Step H: Output results in a JSON file
#     # --------------------------------------------------------------------------------
#     # Each user: 
#     #   {
#     #     "user_name": <string>,
#     #     "files": [
#     #        {"filename": <str>, "file_label": <str>, "pred_label": <str>}, ...
#     #     ],
#     #     "final_vote": <str>    # user-level predicted
#     #   }

#     # Build a map from numeric -> label text for convenience
#     idx2label = {0: "normal", 1: "attack", 2: "triggered"}

#     # We'll use the same user_data_map to build the JSON
#     final_json_output = []
#     for user, info in user_data_map.items():
#         # convert user-level predicted label to string
#         sum_probs = info["prob_sums"]
#         user_pred = np.argmax(sum_probs)
#         user_pred_str = idx2label[user_pred]

#         # file-level details
#         files_info = []
#         for f_label, f_pred, fname in zip(
#                 info["file_labels"], info["file_preds"], info["filenames"]
#         ):
#             files_info.append({
#                 "filename": fname,
#                 "file_label": idx2label[f_label],
#                 "pred_label": idx2label[f_pred]
#             })

#         # add user-level record
#         final_json_output.append({
#             "user_name": user,
#             "files": files_info,
#             "final_vote": user_pred_str
#         })

#     # Save to JSON
#     out_json_path = os.path.abspath(f"../data/guardian/test_user_vote_{name_training}.json")
#     with open(out_json_path, "w", encoding="utf-8") as jf:
#         json.dump({"users": final_json_output}, jf, indent=2)

#     print(f"\nSaved user-based JSON report to: {out_json_path}")

#     # --------------------------------------------------------------------------------
#     # Final timing
#     # --------------------------------------------------------------------------------
#     total_time = tm.time() - start_time
#     print(f"\nTotal inference time: {total_time:.2f} seconds.")


# ################################################################################
# # 5) ENTRY POINT
# ################################################################################
# if __name__ == "__main__":
#     # Example usage:
#     name_training = "8262735442-fold1"

#     # This pattern should point to final 1024-d .npy embeddings (already paired).
#     file_pattern = "../data/sample_dataset/libri_data/embed_test/*.npy"
#     num_of_prediction = 1  # Must be 1, 10, or 20

#     print("Training Model name:", name_training)
#     print("Test files pattern:", file_pattern)
#     print("Number of predictions (DeepSpeaker checkpoints used):", num_of_prediction)
#     # If you still have the array of checkpoint paths in c.CHECKPOINT_FOLDER_ARRAY, you can show it:
#     # print("Deep Speaker checkpoint arrays:\n ", " | ".join(c.CHECKPOINT_FOLDER_ARRAY))

#     main(name_training, file_pattern, num_of_prediction)




###############################################################################
###############################################################################
###############################################################################
###############################################################################

#Workable version with three classes
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################


import glob
import sys
import numpy as np
import tensorflow as tf
import os
import csv
import time as tm
from collections import Counter

# For multi-class metrics:
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append("..")
import guardian.constants as c
from guardian.utils_my_version import (
    auto_stat_test_model,
    get_checkpoint_name_training,
    get_last_checkpoint_if_any,
)

# Load the Deep Speaker model(s)
from authentication_model.deep_speaker_models import convolutional_model


def main(name_training, file_pattern, num_of_prediction):
    """
    Main test function.

    :param name_training: e.g. "1234567890-fold1" – the discriminator checkpoint name suffix.
    :param file_pattern: Glob pattern for the test *.npy files.
    :param num_of_prediction: How many times (and from which 'checkpoint array') to run deep speaker inference.
                              Typically 1, 10, or 20 from your original script.
    """

    # Decide how many times to run deep speaker (the "model1" ensemble) based on num_of_prediction:
    if num_of_prediction == 1:
        deep_speaker_indices = [0]
        times = 1
    elif num_of_prediction == 10:
        deep_speaker_indices = list(range(10))  # 0..9 => 10 total
        times = 10
    elif num_of_prediction == 20:
        deep_speaker_indices = list(range(10)) + list(range(10))  # 20 total
        times = 20
    else:
        raise ValueError("num_of_prediction must be 1, 10, or 20.")

    # Gather test files
    file_list = glob.glob(file_pattern)
    if not file_list:
        print(f"No test files found for pattern: {file_pattern}")
        return

    # Prepare arrays for storing ground truth and predicted labels
    #  - 0 => normal, 1 => attack, 2 => triggered
    y_true = []
    y_pred = []

    # We also create CSV logs.
    # Each line: [filename, ground_truth_str, prob_0, prob_1, prob_2, ..., final_prediction_str]
    csv_rows = []
    header = ["filename", "ground_truth", "checkpoint_probs...(repeated)", "final_pred"]

    print(f"Loading Discriminator model ID from: {name_training}")
    # name_training might look like "9876543210-fold1". The ID is the part before the dash:
    # model_ID = name_training.split("-")[0]
    # If your naming is consistent with the original script, do:
    model_ID = name_training.split("-")[0]

    # 1) Build an array of DeepSpeaker models (model1) – each with the relevant checkpoint
    #    from c.CHECKPOINT_FOLDER_ARRAY. E.g. 10 or 20 of them
    deep_speaker_models = []
    for i in range(times):
        # figure out which array index we’re using
        ds_index = deep_speaker_indices[i]
        dsp_model = convolutional_model()
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER_ARRAY[ds_index])
        if last_checkpoint is not None:
            dsp_model.load_weights(last_checkpoint)
        deep_speaker_models.append(dsp_model)

    # 2) Load the final 3‐class discriminator (model2) from its .h5
    discriminator_path = os.path.join(c.DISCRIMINATOR_MODEL, f"{model_ID}.h5")
    if not os.path.isfile(discriminator_path):
        print(f"ERROR: Discriminator H5 not found at {discriminator_path}.")
        return
    model2 = tf.keras.models.load_model(discriminator_path)

    # Optionally load the best checkpoint for that training if it exists
    best_checkpoint_path = get_checkpoint_name_training(c.DISCRIMINATOR_CHECKPOINT_FOLDER, name_training)
    if best_checkpoint_path is not None:
        model2.load_weights(best_checkpoint_path)
        print(f"Loaded weights from {best_checkpoint_path} for the discriminator model2.")
    else:
        print("No extra checkpoint found for the discriminator. Using the .h5 directly.")

    # ================ LOOP THROUGH TEST FILES =================
    start_time = tm.time()

    for idx, full_path in enumerate(file_list):
        filename = os.path.basename(full_path)

        # Identify ground_truth label from the filename
        #  0 => normal, 1 => attack, 2 => triggered
        if "(" in filename:
            true_label = 1   # attack
            gt_str = "attack"
        elif "[" in filename:
            true_label = 2   # triggered
            gt_str = "triggered"
        else:
            true_label = 0
            gt_str = "normal"

        # We'll accumulate the sum of output probabilities from each checkpoint, then do an average
        sum_probs = np.zeros(3, dtype=float)

        # We'll also store each checkpoint's raw probabilities for CSV debugging
        checkpoint_probs = []

        for speaker_idx in range(times):
            # auto_stat_test_model returns (raw_result[0], "Normal" / "Attack" / "Triggered")
            # but we only need the raw probabilities, which is shape [3].
            raw_prob, _ = auto_stat_test_model(
                deep_speaker_models[speaker_idx],
                model2,
                name_training,
                os.path.dirname(full_path),
                filename,
                speaker_idx
            )
            # raw_prob should be something like [p_normal, p_attack, p_triggered]
            sum_probs += raw_prob
            checkpoint_probs.extend(list(raw_prob))  # for CSV row

        # Average across all "times" deep speaker predictions
        avg_probs = sum_probs / times
        pred_label = np.argmax(avg_probs)
        # Convert to string for final CSV cell
        if pred_label == 0:
            pred_str = "normal"
        elif pred_label == 1:
            pred_str = "attack"
        else:
            pred_str = "triggered"

        # Collect for final metrics
        y_true.append(true_label)
        y_pred.append(pred_label)

        # Build row for CSV
        row = [filename, gt_str] + checkpoint_probs + [pred_str]
        csv_rows.append(row)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(file_list)} files...")

    # =========== WRITE CSV WITH THE RAW PROBABILITIES & FINAL PREDICTIONS ===========
    out_csv = os.path.abspath(f"../data/guardian/knn_model/test_3class_{name_training}.csv")
    # Build a dynamic header that accommodates times × 3 probability columns
    # For example, if times=10 => 10 triplets => 30 columns
    col_for_checkpoints = []
    for i in range(times):
        col_for_checkpoints.append(f"prob_normal_ckpt_{i}")
        col_for_checkpoints.append(f"prob_attack_ckpt_{i}")
        col_for_checkpoints.append(f"prob_triggered_ckpt_{i}")

    header = ["filename", "ground_truth"] + col_for_checkpoints + ["final_pred"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    print(f"\nWrote test predictions/probabilities to CSV: {out_csv}")

    # =========== COMPUTE & PRINT MULTI-CLASS METRICS ===========
    #  y_true, y_pred are numeric [0,1,2]
    labels_str = ["normal", "attack", "triggered"]
    print("\nConfusion Matrix (row=Actual, col=Pred):")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels_str))

    print(f"Total computation time: {tm.time() - start_time:.2f} seconds.")
    return


if __name__ == "__main__":
    # Example usage:
    name_training = "4228480590-fold1"

    # Adjust these as needed:
    file_pattern = "../data/sample_dataset/libri_data/npy/test/*"  # or any glob pattern
    num_of_prediction = 1  # Must be 1, 10, or 20

    print("Training Model name:", name_training)
    print("Test files pattern:", file_pattern)
    print("Number of predictions (DeepSpeaker checkpoints used):", num_of_prediction)
    print("Deep Speaker checkpoint arrays:\n ", " | ".join(c.CHECKPOINT_FOLDER_ARRAY))

    main(name_training, file_pattern, num_of_prediction)
